from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, chunks, top_k=3):
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        top = scored[:top_k]

        return [
            {
                "text": c["text"],
                "source": c["source"],
                "score": float(score)
            }
            for c, score in top
        ]

