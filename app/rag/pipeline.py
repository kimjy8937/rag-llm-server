class RagPipeline:
    def __init__(self, embedder, store, llm):
        self.embedder = embedder
        self.store = store
        self.llm = llm

    def ask(self, query, history=None):
        query_embedding = self.embedder.encode([query])
        results = self.store.search(query_embedding, k=3)

        contexts = [r["text"] for r in results]
        sources = list(set(r["source"] for r in results))

        context_text = "\n".join(contexts)

        prompt = f"""
Answer the question using the context below.

Context:
{context_text}

Question:
{query}
"""

        answer = self.llm.generate(prompt)

        source_text = ", ".join(sources)
        answer_with_source = f"{answer}\n\n[출처: {source_text}]"

        return answer_with_source
