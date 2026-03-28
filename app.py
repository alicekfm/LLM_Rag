'''
STEP 6: Deploy UI using streamlit

Command to run: streamlit run app.py
Command to stop: ctrl C
'''
import streamlit as st
import traceback
from pathlib import Path

st.set_page_config(page_title="Imperial Physics Notes RAG", page_icon="⚛️", layout="wide")

st.title("⚛️Imperial Physics Q&A ")

try:
    # Check files exist
    required = ["rag_answer.py", "chunks.json", "embeddings.npy"]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        st.error(f"Missing required files in this folder: {missing}")
        st.stop()

    # Imports (inside try so we can display errors)
    from sentence_transformers import SentenceTransformer
    from rag_answer import (
        retrieve_chunks,
        build_prompt,
        generate_answer,
        trim_to_last_sentence,
        EMBED_MODEL_NAME,
    )

    @st.cache_resource
    def load_embed_model():
        return SentenceTransformer(EMBED_MODEL_NAME)

    embed_model = load_embed_model()

    query = st.text_input("Your question:", placeholder="e.g. What are Newton's laws of motion?")
    year_label = st.selectbox(
    "Level of explanation (optional)",
    ["Any year","1st year","2nd year","3rd year"]
    )
    year_map={
    "Any year":None,
    "1st year":1,
    "2nd year":2,
    "3rd year":3
    }

    year=year_map[year_label]



    if st.button("Ask", type="primary"):
        q = query.strip()
        if not q:
            st.warning("Write a question first 🙂")
            st.stop()

        with st.spinner("🔎 Retrieving relevant chunks..."):
            retrieved=retrieve_chunks(query,embed_model,year=year)
            top_chunks = [chunk for (chunk, _score) in retrieved]

        with st.spinner("Generating answer..."):
            prompt = build_prompt(q, top_chunks)
            answer = generate_answer(prompt)
            answer = trim_to_last_sentence(answer)

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Sources")
        for i, (chunk, score) in enumerate(retrieved, start=1):
            md = chunk.get("metadata", {}) or {}
            st.markdown(
                f"**[S{i}]** {md.get('subject','?')} | {md.get('filename','?')} | page {md.get('page','?')} | score {score:.3f}"
            )
            with st.expander(f"Show chunk [S{i}] text"):
                st.write(chunk.get("text", ""))

except Exception as e:
    st.error("App crashed — here is the error:")
    st.code(traceback.format_exc())
