# hearthelperapp.py

import streamlit as st
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- AYARLAR ---
GEMINI_API_KEY = "AIzaSyDx1d0jbLlJ0RYBKGw4uWVYjCEiod7pNSI"  # Ekibin API_Key (kendi API_KEY'inizi girebilirsiniz). 

@st.cache_resource(show_spinner="FAISS index loading...")
def load_faiss_and_chunks(index_path="faiss_index.index", chunk_path="faiss_index_chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Gemini API config
genai.configure(api_key=GEMINI_API_KEY)
g_model = genai.GenerativeModel("gemini-pro")

def get_relevant_chunks(question, embed_model, index, chunks, top_k=3):
    q_vec = embed_model.encode([question])
    D, I = index.search(np.array(q_vec).astype("float32"), top_k)
    return [chunks[i] for i in I[0]]

def generate_gemini_answer(question, context_chunks, language):
    if language == "Türkçe":
        prompt = f"""Aşağıdaki metinleri kullanarak kullanıcıdan gelen soruya kısa, bilimsel ve doğru bir cevap ver:

Bağlam:
{chr(10).join(context_chunks)}

Soru: {question}
Cevap:"""
    else:  # English
        prompt = f"""Using the texts below, answer the user's question in a concise, scientific and accurate manner.

Context:
{chr(10).join(context_chunks)}

Question: {question}
Answer:"""
    response = g_model.generate_content(prompt)
    return response.text

# --- STREAMLIT ARAYÜZÜ ---
st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="💬")
lang = st.selectbox("Select language / Dil seçiniz", ["English", "Türkçe"])

if lang == "Türkçe":
    st.markdown("<h1 style='text-align: center;'>Gemini Destekli PDF Soru-Cevap Botu</h1>", unsafe_allow_html=True)
    st.write("Yüklediğimiz PDF'den alınan bilgiyle, sorularınıza Google Gemini ile cevap verir.")
    question = st.text_input("Sorunuzu yazın:", "")
    ask_button = "Sor"
    answer_title = "**Cevap:**"
    spinner_text = "Cevap hazırlanıyor..."
else:
    st.markdown("<h1 style='text-align: center;'>Gemini-based PDF Question-Answer Bot</h1>", unsafe_allow_html=True)
    st.write("Ask any question based on the uploaded PDF. Google Gemini will generate an answer.")
    question = st.text_input("Enter your question:", "")
    ask_button = "Ask"
    answer_title = "**Answer:**"
    spinner_text = "Generating answer..."

if st.button(ask_button):
    with st.spinner(spinner_text):
        index, chunks = load_faiss_and_chunks()
        embed_model = load_embed_model()
        context_chunks = get_relevant_chunks(question, embed_model, index, chunks)
        answer = generate_gemini_answer(question, context_chunks, lang)
    st.markdown(answer_title)
    st.write(answer)