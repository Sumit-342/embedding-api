import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastembed import TextEmbedding

app = FastAPI()

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed_lock = threading.Lock()


class TextsInput(BaseModel):
    texts: list[str]


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/embed")
def embed(input: TextsInput):
    if len(input.texts) > 100:
        raise HTTPException(status_code=400, detail="Max 100 texts per request")

    with embed_lock:
        embeddings = [
            vec.tolist()
            for vec in model.embed(input.texts, batch_size=8)
        ]

    return {
        "count": len(embeddings),
        "embeddings": embeddings,
    }
