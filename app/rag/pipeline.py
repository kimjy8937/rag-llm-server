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

        results = self.store.search(query_embedding, k=10)

        top_results = self.reranker.rerank(query, results, top_k=3)

        contexts = [r["text"] for r in top_results]
        context_text = "\n".join(contexts)

        recent_messages = messages[-4:]
        conversation_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in recent_messages]
        )

        prompt = f"""
    Answer the question using the context below.

    Summary of conversation:
    {summary}

    Recent conversation:
    {conversation_text}

    Context:
    {context_text}

    Question:
    {query}
    """

        answer = self.llm.generate(prompt)

        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": answer})

        if len(messages) > 6:
            new_summary = self.summarize_history(summary, messages[-6:])
            session["summary"] = new_summary
            session["messages"] = messages[-4:]


        sources = []
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
            "sources": sources
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
