from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import FaissStore
from app.llm.hf_llm import HFLlm
from app.rag.pipeline import RagPipeline
from app.api.routes import router
from app.ingestion.document_loader import load_documents_from_folder

app = FastAPI()

# ---------------------------
# 문서 자동 로딩
# ---------------------------
documents = load_documents_from_folder("docs")

print("로드된 문서 수:", len(documents))

# ---------------------------
# RAG 초기화
# ---------------------------
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
