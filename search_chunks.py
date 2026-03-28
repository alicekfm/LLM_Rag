"""
STEP 4: test retrieval
(NUMPY SIMILARITY)
Embeds a user question and computes cosine similarity to your embedded physics chunks,
then returns the top-k most similar chunks with their source + page.

Usage:
python search_chunks.py "derive the density of states in k-space" or whatever query
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_PATH = Path("embeddings.npy")

# chunks.json contains: {id, text, metadata:{source,page,filename,subject}}
CHUNKS_PATH = Path("chunks.json")

TOP_K = 3



def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize. Works for vectors (1D) or matrices (2D)."""
    if mat.ndim == 1:
        denom = max(np.linalg.norm(mat), eps)
        return mat / denom
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return mat / denom


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python search_chunks.py "your question here"')

    query = " ".join(sys.argv[1:]).strip()

    # 1) Load embeddings + chunks
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing {EMBEDDINGS_PATH}. Run embed_chunks.py first.")
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CHUNKS_PATH}. Run chunk_text.py first (it should write chunks.json)."
        )

    embeddings = np.load(EMBEDDINGS_PATH)  # shape: (num_chunks, dim)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Chunks length ({len(chunks)}) != embeddings rows ({embeddings.shape[0]}).\n"
            f"This usually means you re-chunked but didn't re-embed (or vice versa)."
        )

    # 2) Load embedding model + embed the query
    model = SentenceTransformer(MODEL_NAME)
    q_vec = model.encode([query], convert_to_numpy=True)[0]  # shape: (dim,)

    # 3) Cosine similarity = dot product of L2-normalized vectors
    emb_norm = l2_normalize(embeddings)
    q_norm = l2_normalize(q_vec)
    scores = emb_norm @ q_norm  # shape: (num_chunks,)

    # 4) Get top-k indices (highest cosine similarity)
    top_idx = np.argsort(-scores)[:TOP_K]

    print("\n=== QUERY ===")
    print(query)

    print(f"\n=== TOP {TOP_K} CHUNKS ===")
    for rank, idx in enumerate(top_idx, start=1):
        chunk = chunks[int(idx)]
        score = float(scores[int(idx)])

        # The metadata is nested as such: chunk["metadata"]["source"], etc.
        meta = chunk.get("metadata", {})
        source = meta.get("source", meta.get("filename", "unknown_source"))
        page = meta.get("page", "unknown_page")
        subject = meta.get("subject", "unknown_subject")
        filename = meta.get("filename", "unknown_file")

        text = chunk.get("text", "")

        print("\n" + "-" * 80)
        print(
            f"[{rank}] score={score:.4f} | subject={subject} | file={filename} | page={page}\n"
            f"source={source}"
        )
        print("-" * 80)
        print(text)


if __name__ == "__main__":
    main()


"""
Closest k-chunk hyperparameter:

With NumPy: manually compute cosine similarity and sort to get the top-3.
With ChromaDB/FAISS: query the vector index and it returns the top-k.

3 chunks = small, focused context (less noise)
5–8 chunks = safer coverage for complex questions
"""
