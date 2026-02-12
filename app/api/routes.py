from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

get_pipeline = None

# 대화 세션(나중에 Redis로 교체)
chat_sessions = {
    "session1": {
        "messages": [],
        "summary": ""
    }
}


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
        chat_sessions[request.session_id] = {
            "messages": [],
            "summary": ""
        }

    session = chat_sessions[request.session_id]

    with lock:
        response = pipeline.ask(request.question, session)

    return response


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
