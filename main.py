from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat_oziria import repondre
import json

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

from fastapi import Request
from pydantic import BaseModel

class CleRequest(BaseModel):
    cle: str

@app.post("/verifier-cle")
async def verifier_cle(data: CleRequest):
    with open("cles_acces.json", "r") as f:
        cles_valides = json.load(f)
    return {"acces": data.cle in cles_valides}
