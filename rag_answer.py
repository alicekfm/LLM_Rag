"""
STEP 5: RAG answer generation using Hugging Face Inference API

Usage:
python rag_answer.py "what are newton's laws of motion"
python rag_answer.py "what is the generalised uncertainty principle"

HYPERPARAMS: temperature (grounding) and max tokens (length)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Configuration

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

EMBEDDINGS_PATH = Path("embeddings.npy")
CHUNKS_PATH = Path("chunks.json")

TOP_K = 3

# Prefer environment variable:
#   export HF_TOKEN="hf_..."
# Fallback to hardcoded token if env var not set .
HF_TOKEN = os.getenv("HF_TOKEN")

# Adding level filtering by Year of study

SUBJECT_YEAR = {
    "cp": 3,
    "deem": 2,
    "ep": 2,
    "fqm": 3,
    "mech": 1,
    "mm": 2,
    "ow": 1,
    "pp": 3,
    "qm": 2,
    "rel": 1,
    "som": 2,
    "sp": 2,
    "ss": 3,
    "stat": 1,
    "thermo": 2,
    "vfem": 1
}

import re

def trim_to_last_sentence(text: str) -> str:
    """
    If the model gets cut off, trim to the last complete sentence-ish boundary.
    Keeps things readable even with low max_tokens.
    """
    text = text.strip()

    # If it already ends cleanly, keep it
    if re.search(r"[.!?]\s*$", text):
        return text

    # Find last sentence end
    matches = list(re.finditer(r"[.!?](?:\s+|\s*$)", text))
    if matches:
        return text[:matches[-1].end()].strip()

    # Fallback: trim to last newline
    if "\n" in text:
        return text.rsplit("\n", 1)[0].strip()

    return text

import re
from collections import defaultdict
from typing import Optional, List, Dict, Tuple

# Small stopword list, to further extend if needed
STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "to", "in", "and", "or",
    "for", "on", "with", "as", "by", "from", "does", "do", "define", "explain",
    "give", "me", "it", "this", "that", "principle"
}

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())

def _query_ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def route_subjects(
    query: str,
    chunks: List[dict],
    top_n: int = 2,
    min_score: float = 2.0,
    debug: bool = True,
) -> Optional[List[str]]:
    """
    Pure lexical subject router (NO manual mapping).
    Scores each subject by looking for query n-grams inside chunk text.

    Weighting idea:
      - 4-gram hit: +16
      - 3-gram hit: +9
      - 2-gram hit: +4
      - 1-gram token hit: +0.5 each (low weight)
    Then we take the *max chunk score* per subject (strong evidence beats lots of weak noise).

    Returns:
      - list of chosen subjects if confident
      - None if not confident (caller can fallback to all subjects)
    """
    q_tokens = [t for t in _tokenize(query) if t not in STOPWORDS]
    if not q_tokens:
        return None

    # Build n-grams from query (longer first)
    grams_4 = _query_ngrams(q_tokens, 4)
    grams_3 = _query_ngrams(q_tokens, 3)
    grams_2 = _query_ngrams(q_tokens, 2)
    grams_1 = q_tokens[:]  # tokens

    def chunk_score(text_lower: str) -> float:
        score = 0.0
        # Longest matches dominate
        for g in grams_4:
            if g and g in text_lower:
                score += 16.0
        for g in grams_3:
            if g and g in text_lower:
                score += 9.0
        for g in grams_2:
            if g and g in text_lower:
                score += 4.0
        # Single tokens: weak signal
        for t in grams_1:
            if t and t in text_lower:
                score += 0.5
        return score

    # Track best chunk per subject so debug is actionable
    best_by_subject: Dict[str, Tuple[float, dict]] = {}

    for ch in chunks:
        md = ch.get("metadata", {}) or {}
        subj = md.get("subject", "unknown")
        text_lower = (ch.get("text", "") or "").lower()
        s = chunk_score(text_lower)
        if s <= 0:
            continue
        if subj not in best_by_subject or s > best_by_subject[subj][0]:
            best_by_subject[subj] = (s, ch)

    if not best_by_subject:
        if debug:
            print("\n[ROUTER] No lexical matches for query tokens; falling back to all subjects.")
        return None

    ranked = sorted(best_by_subject.items(), key=lambda kv: kv[1][0], reverse=True)

    if debug:
        print("\n[ROUTER] Top subject candidates (lexical):")
        for subj, (s, ch) in ranked[:5]:
            md = ch.get("metadata", {}) or {}
            preview = (ch.get("text", "") or "").strip().replace("\n", " ")[:140]
            print(f"  - {subj:10s} score={s:5.1f}  ({md.get('filename','?')} p{md.get('page','?')})  :: {preview}")

    chosen = [subj for subj, (s, _ch) in ranked[:top_n] if s >= min_score]

    if not chosen:
        if debug:
            print(f"[ROUTER] No subject passed min_score={min_score}; falling back to all subjects.")
        return None

    if debug:
        print(f"[ROUTER] Chosen subjects: {chosen}")
    return chosen


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize. Works for vectors (1D) or matrices (2D)."""
    if mat.ndim == 1:
        denom = max(np.linalg.norm(mat), eps)
        return mat / denom
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return mat / denom


def retrieve_chunks(query: str, embed_model: SentenceTransformer, year: Optional[int] = None, debug: bool = False):
    """
    Returns list of (chunk_dict, score_float) for top-k most similar chunks.

    Year filter:
      - if year is set, we FIRST restrict to that year's subjects (based on SUBJECT_YEAR)
      - subject routing happens WITHIN that restricted pool
      - fallback if routing is not confident or if similarity is weak
    """
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing {EMBEDDINGS_PATH}. Run embed_chunks.py first.")
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_PATH}. Run chunking first to create {CHUNKS_PATH}.")

    embeddings = np.load(EMBEDDINGS_PATH)  # (num_chunks, dim)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks.json length ({len(chunks)}) != embeddings rows ({embeddings.shape[0]}). "
            "You likely embedded a different chunks file than you're searching."
        )

    # attach year to metadata based on subject (and ensure metadata exists)
    for ch in chunks:
        md = ch.get("metadata", {}) or {}
        subject = md.get("subject")
        md["year"] = SUBJECT_YEAR.get(subject)
        ch["metadata"] = md

    # pool by year (optional)
    if year is not None:
        idx_year = [i for i, ch in enumerate(chunks) if (ch.get("metadata", {}) or {}).get("year") == year]
        if debug:
            print(f"[YEAR FILTER] year={year} candidates={len(idx_year)}")
        if not idx_year:
            idx_year = list(range(len(chunks)))
    else:
        idx_year = list(range(len(chunks)))

    # embed query once + full cosine scores (for fallback expansion)
    q_vec = embed_model.encode([query], convert_to_numpy=True)[0]  # (dim,)
    emb_norm_full = l2_normalize(embeddings)
    q_norm = l2_normalize(q_vec)
    scores_full = emb_norm_full @ q_norm  # (N,)

    # subject routing within the year pool
    chunks_year = [chunks[i] for i in idx_year]
    allowed_subjects = route_subjects(query, chunks_year, top_n=2, min_score=2.0, debug=True)

    # build allowed index pool
    if allowed_subjects is None:
        allowed_idx = np.array(idx_year, dtype=int)
    else:
        allowed_idx = np.array(
            [i for i in idx_year if (chunks[i].get("metadata", {}) or {}).get("subject") in allowed_subjects],
            dtype=int
        )
        if allowed_idx.size == 0:
            # fail open to year pool
            allowed_idx = np.array(idx_year, dtype=int)
            allowed_subjects = None

    # rank by semantic similarity within allowed_idx
    scores_subset = scores_full[allowed_idx]
    top_local = np.argsort(-scores_subset)[:TOP_K]
    top_global_idx = allowed_idx[top_local]

    # fallback if similarity is weak (expand to all chunks)
    if top_global_idx.size > 0:
        best_score = float(scores_full[int(top_global_idx[0])])
        if best_score < 0.30:
            if debug:
                print("[FALLBACK] weak match; expanding to ALL chunks.")
            top_global_idx = np.argsort(-scores_full)[:TOP_K]
            allowed_subjects = None
            scores_subset = scores_full[top_global_idx]
            top_local = np.arange(len(top_global_idx))

    results = []
    for gi, li in zip(top_global_idx, top_local):
        chunk = chunks[int(gi)]
        score = float(scores_full[int(gi)])
        results.append((chunk, score))

    print(f"[RETRIEVE] year={year}")
    print(f"[RETRIEVE] allowed_subjects={allowed_subjects}")
    print(f"[RETRIEVE] searching over {len(chunks)} chunks, embeddings shape={embeddings.shape}")
    print(f"[RETRIEVE] q_vec shape={q_vec.shape}, first3={q_vec[:3]}")
    if len(results) > 0:
        all_scores_in_pool = scores_full[allowed_idx] if allowed_idx.size > 0 else scores_full
        print(f"[RETRIEVE] score range (pool): min={float(all_scores_in_pool.min()):.4f}, max={float(all_scores_in_pool.max()):.4f}")

    return results


def build_prompt(question: str, top_chunks: list) -> str:
    """
    top_chunks must be: [chunk_dict, chunk_dict, ...]
    """
    context_blocks = []

    for i, ch in enumerate(top_chunks, start=1):
        md = ch.get("metadata", {}) or {}
        tag = f"S{i}"
        subject = md.get("subject", "unknown")
        filename = md.get("filename", "unknown_file")
        page = md.get("page", "unknown_page")
        source = md.get("source", "unknown_source")

        context_blocks.append(
            f"[{tag} | {subject} | {filename} | p{page} | {source}]\n{ch.get('text','')}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a physics tutor. 
RULES (must follow):
- Answer the question using ONLY the CONTEXT
- Ignore unrelated material in the context.
- If you cannot cite it from the context, write: "I don't know from the provided notes."
-Do NOT invent citations. Use only the tags provided.
-Try to fix the equations that have been garbled and put them in proper Latex form
QUESTION:
{question}

CONTEXT:
{context}

ANSWER (with inline citations):
"""
    return prompt


def print_sources(top_chunks: list):
    print("\n========== SOURCES ==========")
    for i, ch in enumerate(top_chunks, start=1):
        md = ch.get("metadata", {}) or {}
        print(
            f"[S{i}] {md.get('subject','?')} | "
            f"{md.get('filename','?')} | "
            f"page {md.get('page','?')} | "
            f"{md.get('source','?')}"
        )


def generate_answer(prompt: str) -> str:
    if not HF_TOKEN or HF_TOKEN.startswith("hf_") is False:
        # This is a soft check. Your token should start with "hf_".
        raise ValueError(
            "HF_TOKEN is missing/invalid. Set it with:\n"
            '  export HF_TOKEN="hf_..."\n'
            "or put it in the HF_TOKEN variable in this file."
        )

    client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful physics assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.2,
    )

    return response.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python rag_answer.py "your question here"')

    query = " ".join(sys.argv[1:]).strip()

    # Load embedding model once (faster than reloading inside retrieve)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("\n🔎 Retrieving relevant chunks...")
    retrieved = retrieve_chunks(query, embed_model, year=None, debug=True)

    # unpack tuples into chunk dicts for build_prompt()
    top_chunks = [chunk for (chunk, _score) in retrieved]

    print("1️⃣ Building prompt...")
    prompt = build_prompt(query, top_chunks)

    print("2️⃣ Generating answer from LLM...\n")
    answer = generate_answer(prompt)
    answer = trim_to_last_sentence(answer)

    print("========== FINAL ANSWER ==========\n")
    print(answer)

    # Optional: show sources after answer
    print_sources(top_chunks)


if __name__ == "__main__":
    main()