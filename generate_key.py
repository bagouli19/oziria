import uuid
import json
import os

# Fichier où stocker les clés
KEY_FILE = "cles_acces.json"

# Créer le fichier s'il n'existe pas
if not os.path.exists(KEY_FILE):
    with open(KEY_FILE, "w") as f:
        json.dump([], f)

# Générer une clé unique
cle = str(uuid.uuid4())

# Ajouter la clé au fichier
with open(KEY_FILE, "r+") as f:
    cles = json.load(f)
    cles.append(cle)
    f.seek(0)
    json.dump(cles, f, indent=2)

print(f"✅ Nouvelle clé générée : {cle}")