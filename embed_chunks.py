'''
STEP 3: The embedding step converts each chunk of the lecture notes into a numerical vector that captures its semantic
meaning, so similar physics concepts end up close together in vector space.

SentenceTransformers loads a pretrained model from the Hugging Face model hub (for now: all-MiniLM-L6-v2), runs the text through
it, and returns the vectors.

Embedding shape:
(2417, 384) 2417 chunks, and each one has been converted into a 384-dimensional numerical vector (fixed embedding size of the all-MiniLM-L6-v2).

'''

import json
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load the chunked data
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loaded", len(chunks), "chunks")

# Extract just the text for embedding 
texts = [chunk["text"] for chunk in chunks]


# 2. Load embedding model
# This downloads the model the first time from Hugging Face
# After that, it loads from local cache
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded")



# 3. Generate embeddings
# model.encode converts each text string into a vector
# normalize_embeddings=True makes cosine similarity easier later
embeddings = model.encode(
    texts,
    batch_size=64,              # process 64 chunks at a time (faster)
    show_progress_bar=True,     # nice progress bar
    normalize_embeddings=True   # important for cosine similarity
)

print("Embeddings shape:", embeddings.shape)
# This should be (number_of_chunks, 384)



# 4. Save embeddings to disk
# Save vectors as a NumPy file (fast + efficient)
np.save("embeddings.npy", embeddings)
print("Saved embeddings to embeddings.npy")


# 5. Save metadata separately (for retrieval later) #UPDATE: will just use chunks


# We don't want to re-embed again later,
# so we store the original chunk metadata
with open("chunks_metadata.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("Saved metadata to chunks_metadata.json")



 