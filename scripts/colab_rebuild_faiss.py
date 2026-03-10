"""
Colab script — rebuild FAISS index from enlarged corpus (48k chunks, Solon 1024d).

Run on GPU T4 (Runtime > Change runtime type > T4 GPU).

Cell 1 : install deps
Cell 2 : download chunks.jsonl from HF Hub
Cell 3 : encode with Solon + build FAISS
Cell 4 : push FAISS + chunks_map back to HF Hub

Paste each block in a separate Colab cell.
"""

# ── CELL 1 : install ──────────────────────────────────────────────────────────
CELL_1 = """
!pip install -q sentence-transformers faiss-gpu huggingface_hub jsonlines tqdm
"""

# ── CELL 2 : config + download ────────────────────────────────────────────────
CELL_2 = """
import os, json, jsonlines, pathlib, logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

HF_TOKEN   = "hf_XXXXXXXXXXXXXXXXXXXXXXXX"   # <-- colle ton token ici
HF_REPO_ID = "Fascinax/veridicta-index"
MODEL_ID   = "OrdalieTech/Solon-embeddings-large-0.1"

from huggingface_hub import hf_hub_download, HfApi
pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
pathlib.Path("data/index").mkdir(parents=True, exist_ok=True)

print("Downloading chunks.jsonl ...")
hf_hub_download(
    repo_id=HF_REPO_ID, filename="processed/chunks.jsonl",
    repo_type="dataset", token=HF_TOKEN, local_dir="data"
)
with jsonlines.open("data/processed/chunks.jsonl") as r:
    chunks = list(r)
print(f"Loaded {len(chunks)} chunks")
"""

# ── CELL 3 : encode + build FAISS ─────────────────────────────────────────────
CELL_3 = """
import numpy as np, faiss, json, jsonlines
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (!)")

model = SentenceTransformer(MODEL_ID)
model = model.cuda() if torch.cuda.is_available() else model

texts = [c["text"] for c in chunks]
print(f"Encoding {len(texts)} chunks with {MODEL_ID} ...")

BATCH = 256
embeddings = model.encode(
    texts,
    batch_size=BATCH,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)
print(f"Embeddings shape: {embeddings.shape}")   # (48268, 1024)

# Build FAISS IndexFlatIP (cosine via normalized vecs)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype("float32"))
print(f"FAISS index: {index.ntotal} vectors, dim={dim}")

faiss.write_index(index, "data/index/veridicta.faiss")
print("Saved: data/index/veridicta.faiss")

# Rebuild chunks_map (chunk_id -> metadata for retrieval)
with jsonlines.open("data/index/chunks_map.jsonl", mode="w") as w:
    w.write_all(chunks)
print("Saved: data/index/chunks_map.jsonl")

# Save embedding config
cfg = {"model_id": MODEL_ID, "dim": dim, "normalize": True, "ntotal": index.ntotal}
json.dump(cfg, open("data/index/embedding_config.json", "w"))
print("Config:", cfg)
"""

# ── CELL 4 : upload back to HF Hub ────────────────────────────────────────────
CELL_4 = """
from huggingface_hub import HfApi
import os

api = HfApi()

files = {
    "data/index/veridicta.faiss":           "index/veridicta.faiss",
    "data/index/chunks_map.jsonl":          "index/chunks_map.jsonl",
    "data/index/embedding_config.json":     "index/embedding_config.json",
}

for local, remote in files.items():
    size_mb = os.path.getsize(local) / 1e6
    print(f"Uploading {local} ({size_mb:.1f} MB) -> {remote} ...")
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print("  Done.")

print()
print("=== FAISS rebuild complete ===")
print(f"Vectors: {index.ntotal}  |  Dim: {dim}")
print("HF Hub updated: Fascinax/veridicta-index")
print()
print("Next step: local bm25s rebuild")
print("  python -m retrievers.hybrid_rag --build --force")
print("  python -m retrievers.artifacts --upload")
"""

if __name__ == "__main__":
    print("Paste each CELL_N block in a separate Colab cell.")
    print()
    for name, code in [("CELL 1", CELL_1), ("CELL 2", CELL_2),
                        ("CELL 3", CELL_3), ("CELL 4", CELL_4)]:
        print(f"{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(code)
