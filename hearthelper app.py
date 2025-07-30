import streamlit as st
import uuid
import os
import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

st.set_page_config(page_title="HeartHelper", page_icon="hearthelper_logo.png", layout="wide")

# --- Dil Seçimi ve TXT sözlüğü ---
lang = st.sidebar.selectbox("Select Language / Dil Seçiniz", ["Türkçe", "English"])

TXT = {
    "Türkçe": {
        "app_title": "HeartHelper",
        "panel_header": "AI Destekli Kardiyovasküler Sağlık Asistanı",
        "desc": "HeartHelper, kalp ve damar hastalıklarıyla yaşayan bireylerin tedavi süreçlerini destekleyen AI tabanlı bir dijital sağlık asistanıdır.",
        "slogan": "Güvenilir. Kişisel. Yapay Zekâ Destekli.",
        "profile": "Profil Bilgileri",
        "age": "Yaş",
        "gender": "Cinsiyet",
        "disease": "Hastalık",
        "age_opts": ["18-39", "40-64", "65+"],
        "gender_opts": ["Kadın", "Erkek", "Belirtmek istemiyorum"],
        "disease_opts": ["Kalp Yetmezliği", "Hipertansiyon", "Koroner Arter Hastalığı", "Diğer"],
        "info": "• Bilgiler Sağlık Bakanlığı ve TKD rehberlerine dayalıdır.\n• Profilinizi doğru doldurun.\n• Bu uygulama tanı koymaz, acil durumda 112'yi arayın.",
        "features": "• AI tabanlı bilgi erişimi\n• Kişiye özel içerik\n• Kolay ve güvenli kullanım",
        "ask_area": "💬 Soru-Cevap Alanı",
        "question_placeholder": "Sorunuzu yazınız...",
        "ask_button": "Sor",
        "alert": "Acil durumda lütfen en yakın sağlık kurumuna başvurun!",
        "profile_card": "Profil Bilginiz",
        "about": "HeartHelper Hakkında:\n• AI tabanlı, kişisel sağlık asistanı\n• Türkçe soru-cevap desteği",
        "kvkk": "KVKK & GDPR Uyarısı:\nVerileriniz gizlidir, yanıtlar tıbbi tavsiye yerine geçmez.",
        "q": "Soru",
        "a": "Cevap",
        "source": "Kaynak: Sağlık Bakanlığı & TKD 2024",
        "footer": "HeartHelper bir tanı aracı değildir.",
        "copyright": "© 2025 HeartHelper Team",
        "ready_questions": [
            "Kalp krizi riskini azaltmak için nelere dikkat etmeliyim?",
            "Kalp hastaları için önerilen egzersizler nelerdir?",
            "Hipertansiyonlu bir hastanın beslenmesinde nelere dikkat edilmelidir?"
        ],
        "suggestion_card": "💡 Sağlık Önerisi: Düzenli olarak tansiyonunuzu ve kolesterolünüzü kontrol ettirin.",
        "sidebar_login": "Kullanıcı Girişi",
        "sidebar_enter_id": "ID'nizi girin (veya boş bırakın, yeni oluşturulsun):",
        "sidebar_your_id": "Kullanıcı ID'niz:",
        "sidebar_daily_record": "Günlük Sağlık Kaydı",
        "sidebar_bp": "Tansiyon (örn: 120/80):",
        "sidebar_exercise": "Egzersiz (örn: 30dk yürüyüş):",
        "sidebar_medication": "İlaç (örn: Aspirin):",
        "sidebar_save": "Kaydet",
        "sidebar_record_saved": "Günlük kayıt eklendi!"
    },
    "English": {
        "app_title": "HeartHelper",
        "panel_header": "AI-Powered Cardiovascular Health Assistant",
        "desc": "HeartHelper is an AI-powered digital health assistant for people with cardiovascular diseases.",
        "slogan": "Reliable. Personal. AI-Powered.",
        "profile": "Profile Information",
        "age": "Age",
        "gender": "Gender",
        "disease": "Disease",
        "age_opts": ["18-39", "40-64", "65+"],
        "gender_opts": ["Female", "Male", "Prefer not to say"],
        "disease_opts": ["Heart Failure", "Hypertension", "Coronary Artery Disease", "Other"],
        "info": "• Based on Ministry of Health & TSC guidelines.\n• Fill your profile correctly.\n• This app does not diagnose, call emergency in urgent cases.",
        "features": "• AI-based info access\n• Personalized content\n• Easy and safe to use",
        "ask_area": "💬 Q&A Area",
        "question_placeholder": "Type your question...",
        "ask_button": "Ask",
        "alert": "In emergency, please consult a healthcare provider immediately!",
        "profile_card": "Your Profile",
        "about": "About HeartHelper:\n• AI-powered, personal health assistant\n• English Q&A support",
        "kvkk": "Data Privacy (GDPR):\nAll your data is confidential, answers are not medical advice.",
        "q": "Question",
        "a": "Answer",
        "source": "Source: Ministry of Health & TSC 2024",
        "footer": "HeartHelper is not a diagnostic tool.",
        "copyright": "© 2025 HeartHelper Team",
        "ready_questions": [
            "What should I do to reduce the risk of a heart attack?",
            "What exercises are recommended for people with heart disease?",
            "What should hypertensive patients pay attention to in their diet?"
        ],
        "suggestion_card": "💡 Health Tip: Remember to check your blood pressure and cholesterol regularly.",
        "sidebar_login": "User Login",
        "sidebar_enter_id": "Enter your ID (or leave blank to generate):",
        "sidebar_your_id": "Your User ID:",
        "sidebar_daily_record": "Daily Health Record",
        "sidebar_bp": "Blood Pressure (e.g. 120/80):",
        "sidebar_exercise": "Exercise (e.g. 30min walk):",
        "sidebar_medication": "Medication (e.g. Aspirin):",
        "sidebar_save": "Save",
        "sidebar_record_saved": "Record saved!"
    }
}[lang]

# --- Gemini Ayarı ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
g_model = genai.GenerativeModel("models/gemini-1.5-pro")

@st.cache_resource(show_spinner="Yükleniyor... / Loading...")
def load_faiss_and_chunks(index_path="faiss_index.index", chunk_path="faiss_index_chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

@st.cache_resource(show_spinner="Model yükleniyor... / Loading model...")
def load_embed_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_relevant_chunks(question, embed_model, index, chunks, top_k=3):
    q_vec = embed_model.encode([question])
    D, I = index.search(np.array(q_vec).astype("float32"), top_k)
    return [chunks[i] for i in I[0]]

def generate_gemini_answer(question, context_chunks, language):
    if language == "Türkçe":
        prompt = f"""Aşağıdaki metinleri kullanarak kalp ve damar sağlığıyla ilgili gelen soruya bilimsel, anlaşılır ve güvenilir bir cevap ver:\nBağlam:\n{chr(10).join(context_chunks)}\nSoru: {question}\nCevap:"""
    else:
        prompt = f"""Using the texts below, answer the user's question about cardiovascular health in a scientific, clear, and trustworthy way.\nContext:\n{chr(10).join(context_chunks)}\nQuestion: {question}\nAnswer:"""
    response = g_model.generate_content(prompt)
    return response.text

# === Kullanıcı Girişi / ID Sistemi ===
def get_or_create_user_id():
    st.sidebar.markdown(f"#### {TXT['sidebar_login']}")
    id_input = st.sidebar.text_input(TXT['sidebar_enter_id'], "")
    if id_input:
        user_id = id_input.strip()
        st.session_state["user_id"] = user_id
    else:
        if "user_id" not in st.session_state:
            st.session_state["user_id"] = str(uuid.uuid4())[:8]
        user_id = st.session_state["user_id"]
    st.sidebar.markdown(
        f"<div style='color:#38bdf8;font-size:0.93em; text-align:center; margin-top:8px;'>"
        f"{TXT['sidebar_your_id']}<br><b>{user_id}</b></div>",
        unsafe_allow_html=True
    )
    return user_id

user_id = get_or_create_user_id()

# === Sağlık Kayıtları ===
records_dir = "user_records"
os.makedirs(records_dir, exist_ok=True)
datafile = os.path.join(records_dir, f"{user_id}_records.csv")

def load_user_data():
    if os.path.exists(datafile):
        return pd.read_csv(datafile)
    else:
        return pd.DataFrame(columns=["Date", "BloodPressure", "Exercise", "Medication", "Feedback"])

def save_user_data(df):
    df.to_csv(datafile, index=False)

user_data = load_user_data()

st.sidebar.markdown("---")
st.sidebar.markdown(f"#### {TXT['sidebar_daily_record']}")
bp = st.sidebar.text_input(TXT['sidebar_bp'])
ex = st.sidebar.text_input(TXT['sidebar_exercise'])
med = st.sidebar.text_input(TXT['sidebar_medication'])

if st.sidebar.button(TXT['sidebar_save']):
    new_row = {
        "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "BloodPressure": bp,
        "Exercise": ex,
        "Medication": med,
        "Feedback": ""
    }
    user_data = pd.concat([user_data, pd.DataFrame([new_row])], ignore_index=True)
    save_user_data(user_data)
    st.sidebar.success(TXT['sidebar_record_saved'])

# === Profil Bilgileri ===
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown(f"### 👤 {TXT['profile']}")
age = st.sidebar.selectbox(TXT["age"], TXT["age_opts"])
gender = st.sidebar.selectbox(TXT["gender"], TXT["gender_opts"])
disease = st.sidebar.selectbox(TXT["disease"], TXT["disease_opts"])

st.sidebar.markdown("### 📖 Info & Guide" if lang == "English" else "### 📖 Bilgi & Rehber")
st.sidebar.info(TXT["info"])
st.sidebar.markdown(f"#### 💡 {'Features' if lang == 'English' else 'HeartHelper Özellikleri'}")
st.sidebar.success(TXT["features"])
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.info(TXT["suggestion_card"])

# === ANA PANEL ===
colA, colB = st.columns([2, 1])
with colA:
    st.markdown("""
        <style>
            .main-title {font-size:2.3em; color:#60a5fa; font-weight:bold; text-align:left;}
            .desc-card {background:#232946; border-radius:14px; color:#e5e7eb; padding:16px 20px; margin-bottom:16px;}
            .chat-bubble-q {background:#334155; color:#facc15; border-radius:12px; margin-bottom:7px; padding:12px 16px; font-size:1.07em;}
            .chat-bubble-a {background:#222c3a; color:#e5e7eb; border-radius:12px; margin-bottom:11px; padding:14px 18px; font-size:1.10em;}
            .alert-acil {background:#ef4444; color:white; border-radius:10px; padding:9px 12px; margin-bottom:10px; font-weight:bold;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='main-title'>🫀 {TXT['app_title']} Assistant</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='desc-card'>{TXT['desc']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='alert-acil'>{TXT['alert']}</div>", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    # Ön tanımlı sorular
    st.markdown(f"##### {TXT['ask_area']}")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    for i, q in enumerate(TXT["ready_questions"]):
        with [preset_col1, preset_col2, preset_col3][i % 3]:
            # Key'ler benzersiz!
            if st.button(q, key=f"preset_button_{i}", help="Bu soruyu otomatik doldur."):
                st.session_state["inputq"] = q

    question = st.text_input(TXT["question_placeholder"], key="inputq")
    if st.button(TXT["ask_button"], use_container_width=True):
        if question.strip():
            # Gemini ile cevap oluştur!
            with st.spinner("Cevap hazırlanıyor..." if lang == "Türkçe" else "Generating answer..."):
                index, chunks = load_faiss_and_chunks()
                embed_model = load_embed_model()
                context_chunks = get_relevant_chunks(question, embed_model, index, chunks)
                answer = generate_gemini_answer(question, context_chunks, lang)
            st.session_state.history.append({"soru": question, "cevap": answer})

    # Geçmiş Soru-Cevaplar
    for qa in reversed(st.session_state.history):
        st.markdown(f"<div class='chat-bubble-q'><b>{TXT['q']}:</b> {qa['soru']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-a'><b>{TXT['a']}:</b> {qa['cevap']}</div>", unsafe_allow_html=True)

    st.markdown("#### 📋 Günlük Sağlık Kayıtlarınız / Your Daily Health Records")
    if len(user_data):
        st.dataframe(user_data[["Date", "BloodPressure", "Exercise", "Medication"]].sort_values("Date", ascending=False))
    else:
        st.info("Henüz hiç kayıt eklenmedi. / No records have been added yet.")

    # Feedback Alanı
    st.markdown("---")
    st.markdown("### Yanıt Geri Bildirimi / Feedback")
    feedback = st.radio("Bu asistanın verdiği yanıtlar işinize yarıyor mu?", ["Evet / Yes", "Hayır / No", "Kararsızım / Not sure"], index=2)
    if st.button("Feedback Gönder / Send Feedback"):
        if len(user_data):
            user_data.at[user_data.index[-1], "Feedback"] = feedback
            save_user_data(user_data)
            st.success("Geri bildiriminiz kaydedildi, teşekkürler!")

with colB:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("hearthelper_logo.png", width=250, caption="HeartHelper", use_container_width=False)
        st.markdown(f"<div class='desc-card'><b>{TXT['profile_card']}</b><br>"
            f"{TXT['age']}: <b>{age}</b><br>{TXT['gender']}: <b>{gender}</b><br>{TXT['disease']}: <b>{disease}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='desc-card'>{TXT['about']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='desc-card' style='color:#facc15;'>{TXT['kvkk']}</div>", unsafe_allow_html=True)

# --- Footer --- #
st.markdown(
    f"<hr><div style='color:#64748b;font-size:0.92em;text-align:center;margin-top:12px;'>"
    f"{TXT['footer']}<br><b>{TXT['source']}</b><br><b>{TXT['copyright']}</b></div>",
    unsafe_allow_html=True
)
