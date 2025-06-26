import pandas as pd
import os

fichiers = [
    "data/donnees_aapl.csv",
    "data/donnees_tsla.csv",
    "data/donnees_googl.csv",
    "data/donnees_btc-usd.csv",
    "data/donnees_eth-usd.csv"
]

for fichier in fichiers:
    if os.path.exists(fichier):
        df = pd.read_csv(fichier)

        # Si la date est dans l’index
        if 'date' not in df.columns:
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
            elif df.index.name in ['date', 'Date']:
                df.reset_index(inplace=True)
                df.rename(columns={df.columns[0]: 'date'}, inplace=True)
            else:
                # Création d'une colonne date fictive si manquante (à adapter si besoin)
                df.insert(0, 'date', pd.date_range(end=pd.Timestamp.today(), periods=len(df)))
                print(f"⚠️ Colonne 'date' absente : ajout automatique dans {fichier}")

        # Vérification et conversion
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)

        # Sauvegarde
        df.to_csv(fichier, index=False)
        print(f"✅ Fichier corrigé : {fichier}")
    else:
        print(f"❌ Fichier introuvable : {fichier}")
