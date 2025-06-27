from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Charger les variables d'environnement depuis .env
load_dotenv()

# Configuration FastAPI
app = FastAPI()

# Activer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter les fichiers statiques (images, CSS...)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Page d'accueil (formulaire d’accès)
@app.get("/", response_class=HTMLResponse)
async def page_connexion():
    return FileResponse("login.html")  # ← Formulaire avec champ de clé

# Vérification de la clé d'accès
@app.post("/verifier-cle")
async def verifier_cle(request: Request):
    form = await request.form()
    cle_utilisateur = form.get("cle")

    cle_attendue = os.getenv("ACCESS_KEY")

    if cle_utilisateur == cle_attendue:
        return FileResponse("index.html")  # ← Page de chat ou d’interface
    return HTMLResponse(
        "<h1>⛔ Clé invalide</h1><a href='/'>↩️ Retour</a>", status_code=403
    )


