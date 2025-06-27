from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat_oziria import repondre

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
