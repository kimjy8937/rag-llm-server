from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

get_pipeline = None

# 대화 세션(나중에 Redis로 교체)
chat_sessions = {}

class QueryRequest(BaseModel):
    question: str
    session_id: str

@router.post("/ask")
def ask_question(request: QueryRequest):
    if get_pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized")

    pipeline, lock = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline is not ready yet")

    if request.session_id not in chat_sessions:
        chat_sessions[request.session_id] = []
    history = chat_sessions[request.session_id]

    with lock:
        answer = pipeline.ask(request.question, history)

    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": answer})

    return {"answer": answer}

@router.post("/reindex")
def reindex():
    if get_pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized")

    from app.main import build_pipeline

    pipeline, lock = get_pipeline()
    with lock:
        new_pipeline, chunks = build_pipeline()

        import app.main as main
        main.rag_pipeline = new_pipeline

    return {"status": "ok", "indexed_chunks": chunks}
