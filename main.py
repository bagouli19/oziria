from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import urllib.request
from utils.model_loader import load_bert_model 

# Téléchargement du modèle au démarrage si absent
def download_model():
    model_path = "models/bert-base-nli-mean-tokens/flax_model.msgpack"
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("Téléchargement du modèle...")
        urllib.request.urlretrieve(
            "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/flax_model.msgpack",
            model_path
        )

download_model()

# Ensuite seulement, on importe tout le reste
from app.utils.model_loader import load_bert_model
from chat_oziria import repondre   # ← adapte si nom différent

app = FastAPI()
model = load_bert_model()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("index.html")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_endpoint(prompt: PromptRequest):
    response_text = repondre(prompt.prompt)
    return {"response": response_text}
