class RagPipeline:
    def __init__(self, embedder, store, llm):
        self.embedder = embedder
        self.store = store
        self.llm = llm

    def ask(self, query, history):
        query_embedding = self.embedder.encode([query])
        docs = self.store.search(query_embedding, k=2)
        context = "\n".join(docs)

        system_prompt = f"""
You are a helpful chatbot.
Answer only using the context below.

Context:
{context}
"""

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # 이전 대화 기록 추가
        messages.extend(history)

        # 현재 질문 추가
        messages.append({"role": "user", "content": query})

        return self.llm.chat(messages)
