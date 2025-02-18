import os
import time
from fastapi import FastAPI
from pinecone import Pinecone, ServerlessSpec
import uvicorn

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


@app.get("/add-vectors")
async def addVectors():
    try:
        index_name = "quickstart"
        data = [
            {
                "id": "vec1",
                "text": "Apple is a popular fruit known for its sweetness and crisp texture.",
            },
            {
                "id": "vec2",
                "text": "The tech company Apple is known for its innovative products like the iPhone.",
            },
            {
                "id": "vec3",
                "text": "Many people enjoy eating apples as a healthy snack.",
            },
            {
                "id": "vec4",
                "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
            },
            {
                "id": "vec5",
                "text": "An apple a day keeps the doctor away, as the saying goes.",
            },
            {
                "id": "vec6",
                "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership.",
            },
        ]

        accumulated_text = [d["text"] for d in data]

        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=accumulated_text,
            parameters={"input_type": "passage", "truncate": "END"},
        )

        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        index = pc.Index(index_name)

        vectors = []
        for d, e in zip(data, embeddings):
            vectors.append(
                {"id": d["id"], "values": e["values"], "metadata": {"text": d["text"]}}
            )

        index.upsert(vectors=vectors, namespace="ns1")

        return {
            "message": "Vectors added successfully",
        }
    except Exception as e:
        return {"message": "Error adding vectors", "error": str(e)}


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
        query = "Government of India"
        index_name = "quickstart"
        embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query"},
        )
        index = pc.Index(index_name)
        results = index.query(
            namespace="ns1",
            vector=embedding[0].values,
            top_k=1,
            include_values=False,
            include_metadata=True,
        )

        print("results", results)
        return {"results": results.to_dict()}
    except Exception as e:
        return {"message": "Error searching", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
