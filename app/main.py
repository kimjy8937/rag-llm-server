from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import FaissStore
from app.llm.hf_llm import HFLlm
from app.rag.pipeline import RagPipeline
from app.api.routes import router, rag_pipeline


app = FastAPI()

# ---------------------------
# 서버 시작 시 RAG 초기화
# ---------------------------
documents = [
    "Jinyoung is a backend developer.",
    "Jinyoung works at a company called ICTWAY.",
    "He mainly uses Spring Boot and MariaDB.",
    "He is currently building an AI chatbot using RAG.",
    "The chatbot is written in Python with FastAPI."
]

embedder = Embedder()
doc_embeddings = embedder.encode(documents)

store = FaissStore(dimension=doc_embeddings.shape[1])
store.add(doc_embeddings, documents)

llm = HFLlm()
pipeline = RagPipeline(embedder, store, llm)

# 라우터에 pipeline 주입
import app.api.routes as routes
routes.rag_pipeline = pipeline

app.include_router(router)
