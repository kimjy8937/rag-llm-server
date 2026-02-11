from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
import os

from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import FaissStore
from app.llm.hf_llm import HFLlm
from app.rag.pipeline import RagPipeline
from app.api.routes import router
from app.ingestion.document_loader import load_documents_from_folder

app = FastAPI()

embedder = Embedder()

# ---------------------------
# FAISS 인덱스 존재 여부 확인
# ---------------------------
index_path = "index/faiss.index"

if os.path.exists(index_path):
    print("기존 인덱스 사용")
    store = FaissStore(dimension=384)  # all-MiniLM-L6-v2 = 384차원
else:
    print("문서로 새 인덱스 생성")
    documents = load_documents_from_folder("docs")
    print("로드된 문서 수:", len(documents))

    doc_embeddings = embedder.encode(documents)
    store = FaissStore(dimension=doc_embeddings.shape[1])
    store.add(doc_embeddings, documents)

llm = HFLlm()
pipeline = RagPipeline(embedder, store, llm)

import app.api.routes as routes
routes.rag_pipeline = pipeline

app.include_router(router)
