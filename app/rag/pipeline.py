class RagPipeline:
    def __init__(self, embedder, store, llm):
        self.embedder = embedder
        self.store = store
        self.llm = llm

    def ask(self, query, history):
        q_emb = self.embedder.encode([query])
        results = self.store.search(q_emb, k=3)  # [{"text","source"}, ...]

        contexts = [r["text"] for r in results]
        sources = sorted(set(r["source"] for r in results))
        context_text = "\n".join(contexts)

        prompt = f"""Answer the question using the context below.

Context:
{context_text}

Question:
{query}
"""

        answer = self.llm.generate(prompt)
        return f"{answer}\n\n[출처: {', '.join(sources)}]"
