import streamlit as st
import os
import pickle
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import openai
import numpy as np

# Set your OpenAI API key as an environment variable: OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

# Functions for PDF ingestion and chunking
def load_pdf(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Embedding and FAISS index creation
 def embed_text(texts):
    resp = openai.Embedding.create(input=texts, model="text-embedding-3-small")
    return np.array([d["embedding"] for d in resp["data"]])

@st.cache_resource
 def create_faiss_index(chunks):
    embeddings = embed_text(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    # Save index and chunks for later use
    faiss.write_index(index, "faiss.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    return index

@st.cache_resource
 def load_index():
    index = faiss.read_index("faiss.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Retrieve relevant chunks based on question
 def retrieve_chunks(question, index, chunks, k=5):
    q_emb = embed_text([question])
    _, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

# Generate answer using ChatCompletion
 def generate_answer(question, contexts):
    prompt = "Aşağıdaki hasta eğitimi metinlerinden yararlanarak, kalp ve damar hastalarına yönelik nazik ve anlaşılır bir dille yanıt ver:\n\n"
    for i, c in enumerate(contexts):
        prompt += f"Kaynak {i+1}: {c}\n\n"
    prompt += f"Soru: {question}\nCevap:"
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

# Streamlit App
 def main():
    st.set_page_config(page_title="HeartHelper", layout="wide")
    st.title("❤️ HeartHelper - Kalp ve Damar Eğitim Asistanı ❤️")

    # PDF yükleme
    uploaded_file = st.file_uploader("PDF dosyanızı seçin", type="pdf")
    if uploaded_file:
        text = load_pdf(uploaded_file)
        chunks = chunk_text(text)
        st.info("PDF başarıyla yüklendi ve metin chunk'landı!")

        # Index oluştur veya yükle
        if not os.path.exists("faiss.index"):
            index = create_faiss_index(chunks)
            st.success("FAISS index oluşturuldu.")
        else:
            index, chunks = load_index()
            st.success("Mevcut FAISS index yüklendi.")

        # Sohbet arayüzü
        question = st.text_input("Sorunuzu yazın ve Enter'a basın:")
        if question:
            contexts = retrieve_chunks(question, index, chunks)
            answer = generate_answer(question, contexts)
            st.markdown(f"**Cevap:** {answer}")

if __name__ == "__main__":
    main()
