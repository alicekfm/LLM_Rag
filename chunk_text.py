import json
import re
from pathlib import Path
'''
STEP 2: Chunking splits the lecture notes into small, semantically coherent pieces so the model can retrieve only the relevant
parts into the prompt.

Params: split text at paragraph boundaries and group consecutive paragraphs until a maximum character length (~1200 chars) is
reached, so each chunk contains a complete local idea. We also add a small overlap (~200 chars) between chunks to preserve 
context when a concept spans a boundary.
'''


# Splits on blank lines, but keeps paragraphs intact
PARA_SPLIT = re.compile(r"\n\s*\n+")

def iter_paragraphs(text: str):
    text = (text or "").strip()
    if not text:
        return
    for p in PARA_SPLIT.split(text):
        p = p.strip()
        if p:
            yield p

def chunk_paragraphs(paragraphs, chunk_size=1200, chunk_overlap=200):
    """
    Builds chunks approximately by character length.
    Overlap is character overlap, applied between consecutive chunks.
    """
    chunks = []
    cur = ""

    for p in paragraphs:
        # If adding this paragraph would exceed size, flush current chunk
        if cur and len(cur) + 1 + len(p) > chunk_size:
            chunks.append(cur.strip())

            # start next with overlap tail
            if chunk_overlap > 0:
                tail = cur[-chunk_overlap:]
                cur = tail.strip() + "\n" + p
            else:
                cur = p
        else:
            cur = p if not cur else (cur + "\n" + p)

    if cur.strip():
        chunks.append(cur.strip())

    return chunks

def main(in_path="cleaned_text.json", out_path="chunks.json",
         chunk_size=1200, chunk_overlap=200):

    data = json.loads(Path(in_path).read_text(encoding="utf-8"))

    # cleaned_text.json is a LIST of documents
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list of docs, got: {type(data)}")

    out_chunks = []
    chunk_id = 0

    for doc_i, doc in enumerate(data):
        filename = doc.get("filename", f"doc_{doc_i}")
        source = doc.get("source", filename)
        subject = doc.get("subject")

        pages = doc.get("pages", [])
        if not isinstance(pages, list):
            continue

        for page_obj in pages:
            page_num = page_obj.get("page")
            # IMPORTANT: your pages have "text"
            text = page_obj.get("text", "") or ""

            paras = list(iter_paragraphs(text))
            page_chunks = chunk_paragraphs(paras, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            for c in page_chunks:
                out_chunks.append({
                    "id": chunk_id,
                    "text": c,
                    "metadata": {
                        "filename": filename,
                        "source": source,
                        "subject": subject,
                        "page": page_num,
                    }
                })
                chunk_id += 1

    Path(out_path).write_text(json.dumps(out_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_chunks)} chunks to: {out_path}")

if __name__ == "__main__":
    main()
