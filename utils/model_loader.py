import os
from huggingface_hub import snapshot_download

def download_bert_model():
    model_path = os.path.join("models", "bert-base-nli-mean-tokens")
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print("📥 Téléchargement du modèle BERT depuis Hugging Face...")
        snapshot_download(
            repo_id="sentence-transformers/bert-base-nli-mean-tokens",
            local_dir=model_path,
            local_dir_use_symlinks=False,  # évite les problèmes de symlinks
        )
        print("✅ Modèle téléchargé avec succès !")
    else:
        print("✅ Le modèle est déjà présent localement.")

if __name__ == "__main__":
    download_bert_model()