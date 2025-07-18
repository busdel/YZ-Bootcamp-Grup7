# hearthelperapp.py
import streamlit as st
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- AYARLAR ---
GEMINI_API_KEY = ""  # Ekibin API_Key (kendi API_KEY'inizi girebilirsiniz). 

genai.configure(api_key=GEMINI_API_KEY)
g_model = genai.GenerativeModel("models/gemini-1.5-pro")  # veya yeni bir model

st.set_page_config(page_title="HeartHelper", page_icon="🫀", layout="wide")

# ======== DİL SEÇİMİ ========= #
lang = st.sidebar.selectbox("Select Language / Dil Seçiniz", ["Türkçe", "English"])

# ==== DİL TABANLI METİNLER ==== #
def get_profile_fields():
    return ["Yaş", "Cinsiyet", "Hastalık"]

TXT = {
    "Türkçe": {
        "app_title": "HeartHelper",
        "panel_header": "AI Destekli Kardiyovasküler Sağlık Asistanı",
        "desc": "HeartHelper, kalp ve damar hastalıklarıyla yaşayan bireylerin tedavi süreçlerini destekleyen AI tabanlı bir dijital sağlık asistanıdır. Soru-cevaplar bilimsel kaynaklara dayalıdır, kişisel profilinize göre öneriler alırsınız.",
        "profile": "Profil Bilgileri",
        "age": "Yaş",
        "gender": "Cinsiyet",
        "disease": "Hastalık",
        "age_opts": ["18-39", "40-64", "65+"],
        "gender_opts": ["Kadın", "Erkek", "Belirtmek istemiyorum"],
        "disease_opts": ["Kalp Yetmezliği", "Hipertansiyon", "Koroner Arter Hastalığı", "Diğer"],
        "info": "• Bilgiler Sağlık Bakanlığı ve TKD rehberlerine dayalıdır.\n• Kişisel öneriler için profilinizi doğru doldurun.\n• Bu uygulama tanı koymaz, acil durumda 112'yi arayın.",
        "features": "• AI tabanlı bilgi erişimi\n• Bilimsel, kişiye özel içerik\n• Kolay ve güvenli kullanım",
        "ask_area": "💬 Soru-Cevap Alanı",
        "question_placeholder": "Sorunuzu yazınız...",
        "ask_button": "Sor",
        "alert": "Acil durumda lütfen en yakın sağlık kurumuna başvurun!",
        "profile_card": "Profil Bilginiz",
        "profile_fields": ["Yaş", "Cinsiyet", "Hastalık"],
        "about": "HeartHelper Hakkında:\n• AI tabanlı, kişisel sağlık asistanı\n• Kullanıcı dostu, erişilebilir panel\n• Türkçe soru-cevap desteği",
        "kvkk": "KVKK & GDPR Uyarısı:\nTüm verileriniz gizlidir, paylaşılan yanıtlar tıbbi tavsiye yerine geçmez.",
        "q": "Soru",
        "a": "Cevap",
        "source": "Kaynak: Sağlık Bakanlığı & TKD 2024",
        "footer": "HeartHelper bir tanı aracı değildir. Acil durumda sağlık kurumlarına başvurunuz.",
        "copyright": "© 2025 HeartHelper Team"
    },
    "English": {
        "app_title": "HeartHelper",
        "panel_header": "AI-Powered Cardiovascular Health Assistant",
        "desc": "HeartHelper is an AI-powered digital health assistant designed to support individuals with cardiovascular diseases in managing their treatment processes. Q&A are based on scientific sources and personalized by your profile.",
        "profile": "Profile Information",
        "age": "Age",
        "gender": "Gender",
        "disease": "Disease",
        "age_opts": ["18-39", "40-64", "65+"],
        "gender_opts": ["Female", "Male", "Prefer not to say"],
        "disease_opts": ["Heart Failure", "Hypertension", "Coronary Artery Disease", "Other"],
        "info": "• Information is based on the guidelines of the Ministry of Health & Turkish Society of Cardiology.\n• Fill your profile correctly for personalized advice.\n• This app does not diagnose, call emergency in urgent cases.",
        "features": "• AI-based info access\n• Scientific, personalized content\n• Easy and safe to use",
        "ask_area": "💬 Q&A Area",
        "question_placeholder": "Type your question...",
        "ask_button": "Ask",
        "alert": "In emergency, please consult a healthcare provider immediately!",
        "profile_card": "Your Profile",
        "profile_fields": ["Age", "Gender", "Disease"],
        "about": "About HeartHelper:\n• AI-powered, personal health assistant\n• User-friendly, accessible panel\n• English Q&A support",
        "kvkk": "Data Privacy (GDPR):\nAll your data is confidential, answers are not medical advice.",
        "q": "Question",
        "a": "Answer",
        "source": "Source: Ministry of Health & TSC 2024",
        "footer": "HeartHelper is not a diagnostic tool. Please consult a healthcare provider in emergencies.",
        "copyright": "© 2025 HeartHelper Team"
    }
}[lang]

# ==== SIDEBAR ==== #
st.sidebar.markdown("<div style='font-size:54px;text-align:center;'>🫀</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h2 style='color:#60a5fa;text-align:center;'>{TXT['app_title']}</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.markdown(f"### 👤 {TXT['profile']}")
age = st.sidebar.selectbox(TXT["age"], TXT["age_opts"])
gender = st.sidebar.selectbox(TXT["gender"], TXT["gender_opts"])
disease = st.sidebar.selectbox(TXT["disease"], TXT["disease_opts"])

st.sidebar.markdown("### 📖 Info & Guide" if lang == "English" else "### 📖 Bilgi & Rehber")
st.sidebar.info(TXT["info"])
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown(f"#### 💡 {'Features' if lang == 'English' else 'HeartHelper Özellikleri'}")
st.sidebar.success(TXT["features"])

# ===== FAISS ve Model ===== #
@st.cache_resource(show_spinner="Veri tabanı yükleniyor..." if lang == "Türkçe" else "Loading database...")
def load_faiss_and_chunks(index_path="faiss_index.index", chunk_path="faiss_index_chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

@st.cache_resource(show_spinner="Model yükleniyor..." if lang == "Türkçe" else "Loading model...")
def load_embed_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_relevant_chunks(question, embed_model, index, chunks, top_k=3):
    q_vec = embed_model.encode([question])
    D, I = index.search(np.array(q_vec).astype("float32"), top_k)
    return [chunks[i] for i in I[0]]

def generate_gemini_answer(question, context_chunks, age, gender, disease, lang):
    if lang == "Türkçe":
        profile_info = f"Kullanıcının profili: Yaş aralığı: {age}, Cinsiyet: {gender}, Hastalık türü: {disease}."
        prompt = f"""{profile_info}
Aşağıdaki metinleri kullanarak kullanıcıdan gelen soruya kısa, bilimsel, anlaşılır ve güvenilir bir cevap ver. Cevabın sonunda 'Kaynak: Sağlık Bakanlığı ve TKD 2024 Kılavuzları' yaz.

Bağlam:
{chr(10).join(context_chunks)}

Soru: {question}
Cevap:"""
    else:
        profile_info = f"User profile: Age group: {age}, Gender: {gender}, Disease type: {disease}."
        prompt = f"""{profile_info}
Using the texts below, answer the user's question in a concise, scientific, and trustworthy manner. At the end, write 'Source: Ministry of Health & TSC 2024'.

Context:
{chr(10).join(context_chunks)}

Question: {question}
Answer:"""
    response = g_model.generate_content(prompt)
    return response.text

# ==== ANA PANEL / MAIN PANEL ==== #
st.markdown("""
    <style>
        .main-title {font-size:2.3em; color:#60a5fa; font-weight:bold; text-align:left;}
        .desc-card {background:#232946; border-radius:14px; color:#e5e7eb; padding:16px 20px; margin-bottom:16px;}
        .chat-bubble-q {background:#334155; color:#facc15; border-radius:12px; margin-bottom:7px; padding:12px 16px; font-size:1.07em;}
        .chat-bubble-a {background:#222c3a; color:#e5e7eb; border-radius:12px; margin-bottom:11px; padding:14px 18px; font-size:1.10em;}
        .alert-acil {background:#ef4444; color:white; border-radius:10px; padding:9px 12px; margin-bottom:10px; font-weight:bold;}
    </style>
""", unsafe_allow_html=True)

colA, colB = st.columns([2,1])

with colA:
    st.markdown(f"<div class='main-title'>🫀 {TXT['app_title']} Assistant</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='desc-card'>{TXT['desc']}</div>",
        unsafe_allow_html=True
    )
    st.markdown(f"<div class='alert-acil'>{TXT['alert']}</div>", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown(f"##### {TXT['ask_area']}")
    question = st.text_input(TXT["question_placeholder"], key="inputq")

    if st.button(TXT["ask_button"], use_container_width=True):
        if question.strip():
            index, chunks = load_faiss_and_chunks()
            embed_model = load_embed_model()
            context_chunks = get_relevant_chunks(question, embed_model, index, chunks)
            answer = generate_gemini_answer(question, context_chunks, age, gender, disease, lang)
            st.session_state.history.append({"soru": question, "cevap": answer})

    for qa in reversed(st.session_state.history):
        st.markdown(f"<div class='chat-bubble-q'><b>{TXT['q']}:</b> {qa['soru']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-a'><b>{TXT['a']}:</b> {qa['cevap']}</div>", unsafe_allow_html=True)

with colB:
    st.markdown(f"<div class='desc-card'><b>{TXT['profile_card']}</b><br>"
        f"{TXT['age']}: <b>{age}</b><br>{TXT['gender']}: <b>{gender}</b><br>{TXT['disease']}: <b>{disease}</b></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='desc-card'>{TXT['about']}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='desc-card' style='color:#facc15;'>{TXT['kvkk']}</div>",
        unsafe_allow_html=True
    )

# --- Footer --- #
st.markdown(
    f"<hr><div style='color:#64748b;font-size:0.92em;text-align:center;margin-top:12px;'>"
    f"{TXT['footer']}<br><b>{TXT['source']}</b><br><b>{TXT['copyright']}</b></div>",
    unsafe_allow_html=True
)
