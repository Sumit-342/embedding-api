from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastembed import TextEmbedding

app = FastAPI()

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


class TextsInput(BaseModel):
    texts: list[str]


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/embed")
def embed(input: TextsInput):
    if len(input.texts) > 100:
        raise HTTPException(status_code=400, detail="Max 100 texts per request")

    embeddings = [
        vec.tolist()
        for vec in model.embed(input.texts, batch_size=16)
    ]

    return {
        "count": len(embeddings),
        "embeddings": embeddings,
    }
