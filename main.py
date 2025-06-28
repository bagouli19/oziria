from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Charger .env
load_dotenv()

app = FastAPI()

# CORS pour tous
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

# Afficher la page d'accueil (formulaire d'accès)
@app.get("/", response_class=HTMLResponse)
async def page_acces():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Vérification de la clé et redirection vers le chat
@app.post("/verifier-cle", response_class=HTMLResponse)
async def verifier_cle(request: Request):
    form = await request.form()
    cle_utilisateur = form.get("cle")

    cle_attendue = os.getenv("ACCESS_KEY")

    if cle_utilisateur == cle_attendue:
        with open("login.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse("<h1>Clé invalide</h1>", status_code=403)





