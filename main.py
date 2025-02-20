import os
import tempfile
import time
from fastapi import FastAPI, File, UploadFile
from pinecone import Pinecone, ServerlessSpec
import uvicorn
from pdfminer.high_level import extract_text

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

app = FastAPI()


@app.get("/")
async def createIndex():
    try:
        index_name = "quickstart"

        data = pc.create_index(
            name=index_name,
            dimension=1024,  # Replace with your model dimensions
            metric="cosine",  # Replace with your model metric
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created successfully", data)
        return {"message": "Index created successfully"}
    except Exception as e:
        return {"message": "Error creating index", "error": str(e)}


@app.post("/add-vectors")
async def addVectors(file: UploadFile = File(...)):
    try:
        index_name = "quickstart"

        # Read PDF content
        pdf_content = await file.read()

        # Create a temporary file to store PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_content)
            temp_path = temp_file.name

        text = extract_text(temp_path)

        # Clean up temp file
        os.unlink(temp_path)

        # Split text into chunks
        chunks = []
        chunk_size = 500  # Adjust chunk size as needed
        words = text.split()

        for i in range(0, len(words), chunk_size):
            print("words", words[i : i + chunk_size])
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append({"id": f"chunk_{i//chunk_size}", "text": chunk})

        print("CHUNKS", words)
        # Get embeddings for chunks
        accumulated_text = [d["text"] for d in chunks]

        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=accumulated_text,
            parameters={"input_type": "passage", "truncate": "END"},
        )

        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        index = pc.Index(index_name)

        vectors = []
        for d, e in zip(chunks, embeddings):
            vectors.append(
                {"id": d["id"], "values": e["values"], "metadata": {"text": d["text"]}}
            )

        index.upsert(vectors=vectors, namespace="ns2")

        return {
            "message": "PDF vectors added successfully",
            "chunks_processed": len(chunks),
        }
    except Exception as e:
        print(e)
        return {"message": "Error processing PDF and adding vectors", "error": str(e)}


@app.get("/index-stats")
async def indexStats():
    try:
        index_name = "quickstart"
        index = pc.Index(index_name)
        return {"results": str(index.describe_index_stats())}
    except Exception as e:
        return {"message": "Error searching", "error": str(e)}


@app.get("/query")
async def query():
    try:
        query = "what is this story about?"
        index_name = "quickstart"
        embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query"},
        )
        index = pc.Index(index_name)
        results = index.query(
            namespace="ns2",
            vector=embedding[0].values,
            top_k=5,
            include_values=False,
            include_metadata=True,
        )

        return {"results": results.to_dict()}
    except Exception as e:
        return {"message": "Error searching", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
