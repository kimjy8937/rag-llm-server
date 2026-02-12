from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, chunks, top_k=3):

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        # 점수와 함께 정렬
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # top_k만 반환
        return [s[0] for s in scored[:top_k]]
