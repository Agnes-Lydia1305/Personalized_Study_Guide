# app.py

import streamlit as st
import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from google.api_core import retry
import asyncio
import nest_asyncio
nest_asyncio.apply()

# Gemini Retry Setup
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(genai.models.Models.generate_content)

# Load API Key (replace with your key or use secrets manager)
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=GOOGLE_API_KEY)

# PDF Chunking
def load_pdf_chunks(uploaded_file, chunk_size=500):
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Embedding + FAISS
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, embeddings, index

# Query Study Guide
def query_study_guide(topic, chunks, model, index, k=4):
    topic_embedding = model.encode([topic])[0]
    distances, indices = index.search(np.array([topic_embedding]), k)
    retrieved_context = "\n\n".join([chunks[i] for i in indices[0]])

    prompt = f"""
You are an AI tutor. Using only the class notes below, generate a personalized study guide for the topic: "{topic}".

Format the output in JSON like this:
{{
  "topic": "...",
  "summary": "...",
  "key_concepts": ["...", "..."],
  "example_questions": ["...", "..."]
}}

### Class Notes ###
{retrieved_context}
"""

    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=prompt
    )
    return response.text

# Streamlit UI
st.set_page_config(page_title="AI Study Guide Generator", layout="centered")

st.title("📚 Personalized Study Guide Generator (Rust Book Demo)")

uploaded_pdf = st.file_uploader("📄 Upload a PDF (e.g., Rust Book)", type=["pdf"])

if uploaded_pdf:
    with st.spinner("🔍 Processing PDF..."):
        chunks = load_pdf_chunks(uploaded_pdf)
        model, embeddings, index = embed_chunks(chunks)
        st.success("✅ PDF processed and indexed!")

    topic = st.text_input("🧠 Enter a topic/question to generate a study guide:")

    if topic:
        with st.spinner("🤖 Generating personalized study guide..."):
            output = query_study_guide(topic, chunks, model, index)
            st.markdown("### 📌 Study Guide Output")
            st.code(output, language="json")
