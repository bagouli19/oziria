from sentence_transformers import SentenceTransformer

# Téléchargement du modèle BERT localement
model = SentenceTransformer('bert-base-nli-mean-tokens')

print("✅ Modèle BERT téléchargé et prêt à être utilisé.")