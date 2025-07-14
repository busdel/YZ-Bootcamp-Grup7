# hearthelperapp.py
import streamlit as st
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- AYARLAR ---
GEMINI_API_KEY = ""  # Ekibin API_Key (kendi API_KEY'inizi girebilirsiniz). 

# RENK PALETİ VE ÖZEL CSS
st.set_page_config(page_title="HeartHelper – AI Health Assistant", page_icon="❤️", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: #181A20;
        }
        .big-title {
            font-size:2.2em; color:#60a5fa; font-weight:bold; margin-bottom:8px; text-align:center;
            letter-spacing: 0.02em;
        }
        .subtitle {
            font-size:1.3em; color:#10b981; text-align:center; margin-bottom:16px;
        }
        .info-card {
            background: #232946;
            border-radius: 16px;
            padding: 20px 28px;
            margin-bottom: 18px;
            color: #e5e7eb;
            font-weight: 500;
            box-shadow: 0 2px 14px #1e293b99;
        }
        .chat-box {
            background: #222c3a;
            border-radius: 16px;
            padding: 18px 22px;
            box-shadow: 0 2px 16px #60a5fa22;
            font-size: 1.17em;
            color: #e5e7eb;
            margin-bottom: 14px;
        }
        .stTextInput>div>div>input {
            font-size: 1.15em;
            border-radius: 10px;
            border: 2px solid #60a5fa;
            padding: 9px 11px;
            background: #232946;
            color: #e5e7eb;
        }
        .stButton>button {
            border-radius: 12px;
            background-color: #0284c7;
            color: white;
            font-weight: bold;
            font-size: 1.05em;
            padding: 7px 28px;
        }
        .footer {
            color: #a3a3a3;
            font-size: 0.92em;
            text-align: center;
            margin-top: 38px;
        }
        /* SelectBox ve diğer formlar için uyum */
        .stSelectbox>div>div {
            background: #232946;
            color: #e5e7eb;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align:center;font-size:52px;'>🫀</div>", unsafe_allow_html=True)

# === DİL SEÇİMİ ===
lang = st.selectbox("Select Language / Dil Seçiniz", ["Türkçe", "English"])

# === HEADER ===
st.markdown("<div class='big-title'>❤️ HeartHelper</div>", unsafe_allow_html=True)
st.markdown(f"""
    <div class='subtitle'>
    {"AI Destekli Kalp ve Damar Sağlığı Asistanı" if lang=="Türkçe" else "AI-Powered Cardiovascular Health Assistant"}
    </div>
""", unsafe_allow_html=True)

# === ÜRÜN AÇIKLAMASI (KART) ===
desc_tr = """HeartHelper, kalp ve damar hastalıklarıyla yaşayan bireylerin tedavi süreçlerini daha bilinçli, etkili ve kişisel olarak yönetmelerini sağlayan bir yapay zeka destekli sağlık asistanıdır. Bilgiler, Sağlık Bakanlığı ve yetkin medikal otoritelerin rehberlerinden alınır; kullanıcıya sade, doğru ve kişiye özel içerik sunar."""
desc_en = """HeartHelper is an AI-powered digital health assistant designed to empower individuals living with cardiovascular diseases to manage their treatment processes more consciously, effectively, and personally. All information is sourced from official medical guidelines, offering users accurate and personalized content."""

st.markdown(
    f"<div class='info-card'>{desc_tr if lang=='Türkçe' else desc_en}</div>",
    unsafe_allow_html=True
)

# === SORU GİRİŞİ VE SOHBET ===
question = st.text_input(
    "Sorunuzu yazın:" if lang=="Türkçe" else "Type your question:",
    "",
    placeholder=(
        "İlaç kullanımı, egzersiz, beslenme... Kalp sağlığınız için merak ettiklerinizi yazın." 
        if lang=="Türkçe"
        else "Ask anything about heart health, treatment, medication, exercise, or nutrition."
    )
)
ask_button = "Sor" if lang=="Türkçe" else "Ask"

# === Gemini AI ve FAISS yükleme ===
@st.cache_resource(show_spinner="Yükleniyor... / Loading...")
def load_faiss_and_chunks(index_path="faiss_index.index", chunk_path="faiss_index_chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

@st.cache_resource(show_spinner="Model yükleniyor... / Loading model...")
def load_embed_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

genai.configure(api_key=GEMINI_API_KEY)
g_model = genai.GenerativeModel("models/gemini-1.5-pro") # kararlı model olduğu için bu tercih edildi. 

def get_relevant_chunks(question, embed_model, index, chunks, top_k=3):
    q_vec = embed_model.encode([question])
    D, I = index.search(np.array(q_vec).astype("float32"), top_k)
    return [chunks[i] for i in I[0]]

def generate_gemini_answer(question, context_chunks, language):
    if language == "Türkçe":
        prompt = f"""Aşağıdaki metinleri kullanarak kalp ve damar sağlığıyla ilgili gelen soruya bilimsel, anlaşılır ve güvenilir bir cevap ver:

Bağlam:
{chr(10).join(context_chunks)}

Soru: {question}
Cevap:"""
    else:
        prompt = f"""Using the texts below, answer the user's question about cardiovascular health in a scientific, clear, and trustworthy way.

Context:
{chr(10).join(context_chunks)}

Question: {question}
Answer:"""
    response = g_model.generate_content(prompt)
    return response.text

# === SOHBET VE CEVAP ===
if st.button(ask_button):
    with st.spinner("Cevap hazırlanıyor..." if lang=="Türkçe" else "Generating answer..."):
        index, chunks = load_faiss_and_chunks()
        embed_model = load_embed_model()
        context_chunks = get_relevant_chunks(question, embed_model, index, chunks)
        answer = generate_gemini_answer(question, context_chunks, lang)
    st.markdown(
        f"<div class='chat-box'><b>{'Cevap:' if lang=='Türkçe' else 'Answer:'}</b><br>{answer}</div>",
        unsafe_allow_html=True
    )

# === EK ÖZELLİK VE AÇIKLAMA KARTI ===
if lang == "Türkçe":
    st.markdown("""
    <div class='info-card'>
    <b>HeartHelper Özellikleri:</b><br>
    • Yapay zeka destekli akıllı bilgi erişimi<br>
    • Bilimsel kaynaklara dayalı güvenilir içerik<br>
    • Kişiselleştirilmiş öneri ve hatırlatmalar<br>
    • Sade, erişilebilir ve hasta dostu arayüz<br>
    • KVKK/GDPR uyumlu veri güvenliği<br>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='info-card'>
    <b>HeartHelper Features:</b><br>
    • AI-powered smart information access<br>
    • Reliable, guideline-based content<br>
    • Personalized recommendations and reminders<br>
    • Simple, accessible, and patient-friendly interface<br>
    • Data privacy (GDPR-compliant)<br>
    </div>
    """, unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
    <div class='footer'>
    HeartHelper is not a substitute for medical diagnosis or direct consultation. 
    <br>Powered by AI – Designed for safer, smarter self-care.
    <br><br>
    <b>© 2025 HeartHelper Team</b>
    </div>
""", unsafe_allow_html=True)
