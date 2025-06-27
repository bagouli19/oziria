from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
import os

templates = Jinja2Templates(directory="templates")


# Charger les variables d'environnement depuis .env
load_dotenv()

# Configuration FastAPI + CORS
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")

# Page d'accueil
@app.get("/", response_class=HTMLResponse)
async def accueil():
    return FileResponse("index.html")

# Vérification de la clé
@app.post("/verifier-cle")
async def verifier_cle(request: Request):
    form = await request.form()
    cle_utilisateur = form.get("cle")

    cle_attendue = os.getenv("ACCESS_KEY")
    if cle_utilisateur == cle_attendue:
        return FileResponse("index.html")
    return HTMLResponse("<h1>Clé invalide</h1>", status_code=403) 

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


