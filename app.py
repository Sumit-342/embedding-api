from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/embed")
def embed(input: TextInput):
    vector = model.encode(input.text).tolist()
    return {"embedding": vector}
