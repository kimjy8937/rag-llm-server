from app.reranker.reranker import Reranker


class RagPipeline:
    def __init__(self, embedder, store, llm):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.reranker = Reranker()

    def ask(self, query, session):
        messages = session["messages"]
        summary = session["summary"]

        query_embedding = self.embedder.encode([query])

        # 1차 검색
        results = self.store.search(query_embedding, k=10)

        # 2차 정렬
        top_results = self.reranker.rerank(query, results, top_k=3)

        # 최고 점수 확인
        best_score = top_results[0]["score"] if top_results else 0

        USE_DOC_THRESHOLD = 0.3

        use_document = best_score >= USE_DOC_THRESHOLD

        if use_document:
            contexts = [r["text"] for r in top_results]
            context_text = "\n".join(contexts)

            system_instruction = """
    Answer ONLY using the provided context.
    If the answer is not in the context, say you don't know.
    """
        else:
            context_text = ""
            system_instruction = """
    The question is not related to the documents.
    Answer using your general knowledge.
    """

        # 최근 대화
        recent_messages = messages[-4:]
        conversation_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in recent_messages]
        )

        prompt = f"""
    {system_instruction}

    Summary:
    {summary}

    Recent conversation:
    {conversation_text}

    Context:
    {context_text}

    Question:
    {query}
    """

        answer = self.llm.generate(prompt)

        # 메시지 저장
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": answer})

        # 요약 갱신
        if len(messages) > 6:
            session["summary"] = self.summarize_history(summary, messages[-6:])
            session["messages"] = messages[-4:]

        # sources 구성
        sources = []
        if use_document:
            seen = set()
            for r in top_results:
                key = (r["source"], r["text"])
                if key not in seen:
                    sources.append({
                        "file": r["source"],
                        "snippet": r["text"][:200]
                    })
                    seen.add(key)

        return {
            "answer": answer,
            "sources": sources,
            "mode": "document" if use_document else "general"
        }

    def summarize_history(self, summary, messages):

        conversation = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages]
        )

        prompt = f"""
    Summarize the following conversation briefly.

    Previous summary:
    {summary}

    New conversation:
    {conversation}

    Updated summary:
    """

        return self.llm.generate(prompt)
