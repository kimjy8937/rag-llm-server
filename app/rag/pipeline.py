class RagPipeline:
    def __init__(self, embedder, store, llm):
        self.embedder = embedder
        self.store = store
        self.llm = llm

    def ask(self, query, history):
        query_embedding = self.embedder.encode([query])
        results = self.store.search(query_embedding, k=3)

        contexts = [r["text"] for r in results]
        context_text = "\n".join(contexts)

        prompt = f"""
Answer the question using the context below.

Context:
{context_text}

Question:
{query}
"""

        answer = self.llm.generate(prompt)

        # sources 구성
        sources = []
        seen = set()

        for r in results:
            key = (r["source"], r["text"])
            if key not in seen:
                sources.append({
                    "file": r["source"],
                    "snippet": r["text"][:200]  # 최대 200자
                })
                seen.add(key)

        return {
            "answer": answer,
            "sources": sources
        }
