from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    session_id: str


rag_pipeline = None

# 메모리 기반 대화 기록 저장소
chat_sessions = {}

@router.post("/ask")
def ask_question(request: QueryRequest):
    session_id = request.session_id

    # 세션 없으면 생성
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    history = chat_sessions[session_id]

    # RAG 호출
    answer = rag_pipeline.ask(request.question, history)

    # 대화 기록 저장
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": answer})

    return {"answer": answer}
