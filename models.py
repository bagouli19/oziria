from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="sentence-transformers/bert-base-nli-mean-tokens",
    cache_dir="models/bert-base-nli-mean-tokens",
    library_name="sentence_transformers"
)