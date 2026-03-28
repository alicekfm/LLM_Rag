'''
STEP 1: This code runs through all the physics PDFs, extracts the text page by page, and saves it together with metadata 
(subject, filename, page number) into a JSON file for later use.
'''

import pypdf
import os
from pathlib import Path
import json

# 1. Setup
pdf_folder = Path("Physics BSc Notes")
all_documents = []

# 2. Walk through subject folders
for subject_folder in pdf_folder.iterdir():
    if not subject_folder.is_dir():
        continue

    subject = subject_folder.name

    for pdf_path in subject_folder.glob("*.pdf"):
        reader = pypdf.PdfReader(str(pdf_path))

        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                pages.append({
                    "page": page_num,
                    "text": text.strip()
                })

        if not pages:
            continue

        all_documents.append({
            "subject": subject,
            "filename": pdf_path.name,
            "source": str(pdf_path),
            "pages": pages
        })

# 3. Save extracted text
with open("extracted_texts.json", "w", encoding="utf-8") as f:
    json.dump(all_documents, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(all_documents)} PDFs")
for doc in all_documents:
    print(f"  - {doc['subject']}/{doc['filename']} ({len(doc['pages'])} pages)")
