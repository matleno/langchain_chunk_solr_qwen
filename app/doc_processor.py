import re
import time
import pysolr
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from app.prompt import query_chain, sintesi_chain, rag_chain
import os
from dotenv import load_dotenv

load_dotenv()

SOLR_ADDRESS = os.getenv("SOLR_ADDRESS", "http://localhost:8983/solr/pc")
TOP_K = int(os.getenv("TOP_K", "35"))
TOP_K_CHUNK = int(os.getenv("TOP_K_CHUNK", "12"))
TOKEN_MAX = int(os.getenv("TOKEN_MAX", "4500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
SINTESI = int(os.getenv("SINTESI", "False"))

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", 
    trust_remote_code=True, 
    local_files_only=False
)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Solr e modello per embeddings
solr = pysolr.Solr(SOLR_ADDRESS, always_commit=True)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def clean_text(text: str) -> str:
    text = re.sub(r'-\s*[\r\n]+', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_documenti(text: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.create_documents([text])

def generate_query(user_question: str) -> str:
    generated_query = query_chain.invoke({"question": user_question})
    # query_chain restituisce query:""
    generated_query = re.sub(r'^(query:|q:)\s*', '', generated_query, flags=re.IGNORECASE)
    return generated_query.strip().strip('"')

def process_documents(question: str, query: str, 
                      top_k=TOP_K,
                      top_k_chunk=TOP_K_CHUNK,
                      token_max=TOKEN_MAX,
                      use_sintesi=False):
    print(f"*** Avvio per: {query} - doc da solr -> {top_k} chunk -> {top_k_chunk} ***")

    query_embedding = model.encode([query])
    # Query Solr (knn)
    results = solr.search(
        fl=['ID', 'Testo', 'vector', 'score'],
        q=f"{{!knn f=vector topK={top_k}}}{[float(w) for w in query_embedding[0]]}",
        fq="Fonte:Comuni",
        rows=top_k
    )

    print(f"Documenti trovati: {len(results)}")
    all_chunks = []
    for idx, doc in enumerate(results, 1):
        full_text = clean_text(doc.get("Testo", ""))
        print("\n**** Documento numero ", idx, ": ", full_text)
        print("Documento solr numero", idx, ":","-"*50 ,"\n", full_text, "\n", "-"*50)	
        small_docs = chunk_documenti(full_text)
        
        # embedding chunk
        for chunk_doc in small_docs:
            chunk_text_str = chunk_doc.page_content
            chunk_embedding = model.encode([chunk_text_str])[0]
            sim = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            all_chunks.append((chunk_text_str, sim))

    # Ordina e seleziona
    all_chunks.sort(key=lambda x: x[1], reverse=True)

    while top_k_chunk > 0:
        top_chunks = all_chunks[:top_k_chunk]
        final_context = []
        
        for i, (chunk_text_str, sim) in enumerate(top_chunks, start=1):
            print(f"\n--- Chunk {i} ---")
            print(f"Similarity: {sim}")
            print(f"Lunghezza: {len(chunk_text_str)}")
            print(f"testo: {chunk_text_str!r}")
            if i == 5: # visualizza 5 chunk
                break

        # Se "sintesi"==True, sintetizza ogni chunk, altrimenti usa il chunk così com'è
        for i, (chunk_str, sim) in enumerate(top_chunks):
            if use_sintesi:
                summary = sintesi_chain.invoke({"document": chunk_str})
                final_context.append(f"DOCUMENTO {i+1}:\n{summary.strip()}")
            else:
                final_context.append(f"DOCUMENTO {i+1}:\n{chunk_str.strip()}")

        contesto = "\n\n".join(final_context)
        token_count = count_tokens(contesto)

        if token_count > token_max:
            # riduci chunk finché non rientri nei limiti
            top_k_chunk -= 1
        else:
            # Genera la risposta
            start = time.time()
            risposta = rag_chain.invoke({
                "document": contesto,
                "question": question
            })
            end = time.time()
            print(f"Tempo impiegato: {end - start} secondi")
            return risposta.strip() 

    return "ERRORE, Non posso rispondere."