from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding

app = FastAPI()

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/embed")
def embed(input: TextInput):
    vector = list(model.embed([input.text]))[0].tolist()
    return {"embedding": vector}
