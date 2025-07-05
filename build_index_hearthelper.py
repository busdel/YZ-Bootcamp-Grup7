# build_index.py
#Çalıştırmadan önce gerekli kütüphaneleri yükle. 
#pip install streamlit sentence-transformers faiss-cpu PyMuPDF langchain google-generativeai
#1️. Embedding ve FAISS index oluşturma scripti
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def embed_text(texts):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index

def save_faiss_index(index, chunks, path="faiss_index"):
    faiss.write_index(index, f"{path}.index")
    with open(f"{path}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    pdf_path = "all_pdf_bootcamp.pdf"
    print("PDF okunuyor...")
    text = extract_text_from_pdf(pdf_path)
    print("Chunk'lanıyor...")
    chunks = chunk_text(text)
    print(f"{len(chunks)} adet chunk oluşturuldu.")
    print("Embedding'ler alınıyor...")
    embeddings = embed_text(chunks)
    print("FAISS index oluşturuluyor...")
    index = build_faiss_index(np.array(embeddings))
    print("Kaydediliyor...")
    save_faiss_index(index, chunks)
    print("Bilgi tabanı kaydedildi.")