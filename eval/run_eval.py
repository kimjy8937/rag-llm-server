import argparse
import json
import os
from pathlib import Path

from app.ingestion.document_loader import load_documents_from_folder
from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import FaissStore
from app.reranker.reranker import Reranker


def load_cases(path: str):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def ensure_empty_dir(dir_path: str):
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    # 평가 인덱스는 매번 새로 만들도록 기존 파일 삭제
    for name in ["faiss.index", "docs.pkl"]:
        fp = p / name
        if fp.exists():
            fp.unlink()


def build_eval_index(docs_dir: str, index_dir: str):
    ensure_empty_dir(index_dir)

    chunks = load_documents_from_folder(docs_dir)
    if not chunks:
        raise RuntimeError(f"No documents loaded from {docs_dir}")

    texts = [c["text"] for c in chunks]

    embedder = Embedder()
    embeddings = embedder.encode(texts)

    store = FaissStore(
        dimension=embeddings.shape[1],
        index_path=str(Path(index_dir) / "faiss.index"),
        meta_path=str(Path(index_dir) / "docs.pkl"),
    )
    store.add(embeddings, chunks)

    reranker = Reranker()
    return embedder, store, reranker


def safe_get_score(item):
    # reranker가 score를 포함하지 않는 버전이어도 동작하도록 방어
    if isinstance(item, dict) and "score" in item:
        return float(item["score"])
    return None


def evaluate(
    cases,
    embedder,
    store,
    reranker,
    candidate_k: int,
    rerank_k: int,
    threshold: float,
):
    total_doc = 0
    hit1 = hit3 = hit5 = 0
    mrr_sum = 0.0

    total_mode = 0
    mode_correct = 0

    misses = []

    for c in cases:
        q = c["question"]
        expected_sources = set(c.get("expected_sources", []))
        expected_mode = c.get("expected_mode", "document")

        # 1) 벡터 검색 후보
        q_emb = embedder.encode([q])
        candidates = store.search(q_emb, k=candidate_k)  # [{"text","source"}, ...]

        # 2) rerank
        reranked = reranker.rerank(q, candidates, top_k=rerank_k)

        ranked_sources = []
        for r in reranked:
            # r이 dict 형태일 것으로 가정(우리 프로젝트 기준)
            if isinstance(r, dict):
                ranked_sources.append(r.get("source"))
            else:
                # 혹시 문자열/다른 타입이면 무시
                ranked_sources.append(None)

        # 3) mode 예측(문서 사용 vs general) - 최고 rerank 점수 기반
        predicted_mode = "document"
        best_score = None
        if reranked:
            best_score = safe_get_score(reranked[0])
            if best_score is not None and best_score < threshold:
                predicted_mode = "general"

        # mode accuracy
        total_mode += 1
        if predicted_mode == expected_mode:
            mode_correct += 1

        # 문서기반 케이스만 retrieval metric 계산
        if expected_sources:
            total_doc += 1

            # hit@k
            top1 = set(ranked_sources[:1])
            top3 = set(ranked_sources[:3])
            top5 = set(ranked_sources[:5])

            h1 = len(top1 & expected_sources) > 0
            h3 = len(top3 & expected_sources) > 0
            h5 = len(top5 & expected_sources) > 0

            hit1 += 1 if h1 else 0
            hit3 += 1 if h3 else 0
            hit5 += 1 if h5 else 0

            # MRR
            rr = 0.0
            for i, src in enumerate(ranked_sources):
                if src in expected_sources:
                    rr = 1.0 / (i + 1)
                    break
            mrr_sum += rr

            # miss 기록
            if not h5:
                misses.append({
                    "id": c.get("id"),
                    "question": q,
                    "expected": sorted(expected_sources),
                    "got_top5": ranked_sources[:5],
                    "best_score": best_score,
                })

    results = {
        "doc_cases": total_doc,
        "recall@1": (hit1 / total_doc) if total_doc else 0.0,
        "recall@3": (hit3 / total_doc) if total_doc else 0.0,
        "recall@5": (hit5 / total_doc) if total_doc else 0.0,
        "mrr": (mrr_sum / total_doc) if total_doc else 0.0,
        "mode_accuracy": (mode_correct / total_mode) if total_mode else 0.0,
        "total_cases": len(cases),
        "misses": misses,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", default="docs")
    parser.add_argument("--cases", default="eval/eval_cases.jsonl")
    parser.add_argument("--index_dir", default="index_eval")
    parser.add_argument("--candidate_k", type=int, default=20)
    parser.add_argument("--rerank_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    embedder, store, reranker = build_eval_index(args.docs_dir, args.index_dir)
    cases = load_cases(args.cases)

    results = evaluate(
        cases=cases,
        embedder=embedder,
        store=store,
        reranker=reranker,
        candidate_k=args.candidate_k,
        rerank_k=args.rerank_k,
        threshold=args.threshold,
    )

    print("=== RAG Retrieval Eval ===")
    print(f"Total cases: {results['total_cases']}")
    print(f"Doc cases:   {results['doc_cases']}")
    print(f"Recall@1:    {results['recall@1']:.3f}")
    print(f"Recall@3:    {results['recall@3']:.3f}")
    print(f"Recall@5:    {results['recall@5']:.3f}")
    print(f"MRR:         {results['mrr']:.3f}")
    print(f"Mode acc:    {results['mode_accuracy']:.3f}")

    if results["misses"]:
        print("\n--- Misses (expected not in top5) ---")
        for m in results["misses"][:10]:
            print(f"- {m['id']}: {m['question']}")
            print(f"  expected: {m['expected']}")
            print(f"  got_top5: {m['got_top5']}")
            print(f"  best_score: {m['best_score']}")
    else:
        print("\nNo misses in top5. Nice!")


if __name__ == "__main__":
    main()
