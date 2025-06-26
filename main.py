from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat_oziria import repondre   # ← adapte si nom différent
import os
from app.utils.model_loader import load_bert_model
import urllib.request
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = load_bert_model()

# CORS (HTML et API même origine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fichiers statiques (logo, css…)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Page d’accueil
@app.get("/")
def read_index():
    return FileResponse("index.html")

# Schéma de requête
class PromptRequest(BaseModel):
    prompt: str

# Endpoint Chat
@app.post("/chat")
async def chat_endpoint(prompt: PromptRequest):
    response_text = repondre(prompt.prompt)
    return {"response": response_text}

def download_model():
    model_path = "models/bert-base-nli-mean-tokens/flax_model.msgpack"
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("Téléchargement du modèle...")
        urllib.request.urlretrieve(
            "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/flax_model.msgpack",
            model_path
        )