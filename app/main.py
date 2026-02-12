from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
import threading

from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import FaissStore
from app.llm.hf_llm import HFLlm
from app.rag.pipeline import RagPipeline
from app.ingestion.document_loader import load_documents_from_folder
from app.api.routes import router

app = FastAPI()

pipeline_lock = threading.Lock()
rag_pipeline = None

def build_pipeline():
    # 1) 문서 로드
    docs = load_documents_from_folder("docs")
    if not docs:
        raise RuntimeError("docs 폴더에서 로드된 문서가 없습니다.")

    # 2) 임베딩 대상 텍스트만 뽑기
    texts = [d["text"] for d in docs]

    # 3) 임베딩 생성
    embedder = Embedder()
    embeddings = embedder.encode(texts)

    # 4) 벡터스토어 구성
    store = FaissStore(dimension=embeddings.shape[1])
    store.add(embeddings, docs)  # docs는 {"text","source"} 리스트

    # 5) LLM
    llm = HFLlm()

    # 6) Pipeline
    return RagPipeline(embedder, store, llm), len(docs)

@app.on_event("startup")
def startup_event():
    global rag_pipeline
    with pipeline_lock:
        rag_pipeline, n = build_pipeline()
        print(f"[startup] indexed chunks: {n}")


import app.api.routes as routes
routes.get_pipeline = lambda: (rag_pipeline, pipeline_lock)

app.include_router(router)
