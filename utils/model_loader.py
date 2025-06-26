import os
import urllib.request
from sentence_transformers import SentenceTransformer

MODEL_PATH = os.path.join("models", "bert-base-nli-mean-tokens")


def download_model():
    model_file = os.path.join(MODEL_PATH, "flax_model.msgpack")
    if not os.path.exists(model_file):
        os.makedirs(MODEL_PATH, exist_ok=True)
        print("📥 Téléchargement du modèle BERT…")
        urllib.request.urlretrieve(
            "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/flax_model.msgpack",
            model_file
        )


def load_bert_model():
    download_model()
    return SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")