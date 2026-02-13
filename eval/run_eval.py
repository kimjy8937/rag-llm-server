import argparse
import json
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
    if isinstance(item, dict) and "score" in item:
        return float(item["score"])
    return None


def init_bucket():
    return {
        "total": 0,
        "hit1": 0,
        "hit3": 0,
        "hit5": 0,
        "mrr_sum": 0.0,
    }


def update_bucket(bucket, ranked_sources, expected_sources):
    bucket["total"] += 1

    top1 = set(ranked_sources[:1])
    top3 = set(ranked_sources[:3])
    top5 = set(ranked_sources[:5])

    h1 = len(top1 & expected_sources) > 0
    h3 = len(top3 & expected_sources) > 0
    h5 = len(top5 & expected_sources) > 0

    bucket["hit1"] += 1 if h1 else 0
    bucket["hit3"] += 1 if h3 else 0
    bucket["hit5"] += 1 if h5 else 0

    rr = 0.0
    for i, src in enumerate(ranked_sources):
        if src in expected_sources:
            rr = 1.0 / (i + 1)
            break
    bucket["mrr_sum"] += rr

    return h5, rr


def finalize_bucket(bucket):
    total = bucket["total"]
    if total == 0:
        return {
            "count": 0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0,
        }

    return {
        "count": total,
        "recall@1": bucket["hit1"] / total,
        "recall@3": bucket["hit3"] / total,
        "recall@5": bucket["hit5"] / total,
        "mrr": bucket["mrr_sum"] / total,
    }


def evaluate(
    cases,
    embedder,
    store,
    reranker,
    candidate_k: int,
    rerank_k: int,
    threshold: float,
):
    global_bucket = init_bucket()
    by_source = {}
    by_tag = {}

    total_mode = 0
    mode_correct = 0

    misses = []

    for c in cases:
        q = c["question"]
        expected_sources = set(c.get("expected_sources", []))
        expected_mode = c.get("expected_mode", "document")
        tags = c.get("tags", [])

        q_emb = embedder.encode([q])
        candidates = store.search(q_emb, k=candidate_k)
        reranked = reranker.rerank(q, candidates, top_k=rerank_k)

        ranked_sources = []
        for r in reranked:
            if isinstance(r, dict):
                ranked_sources.append(r.get("source"))
            else:
                ranked_sources.append(None)

        predicted_mode = "document"
        best_score = None
        if reranked:
            best_score = safe_get_score(reranked[0])
            if best_score is not None and best_score < threshold:
                predicted_mode = "general"

        total_mode += 1
        if predicted_mode == expected_mode:
            mode_correct += 1

        if expected_sources:
            hit5, _ = update_bucket(global_bucket, ranked_sources, expected_sources)

            for src in expected_sources:
                if src not in by_source:
                    by_source[src] = init_bucket()
                update_bucket(by_source[src], ranked_sources, expected_sources)

            for tag in tags:
                if tag not in by_tag:
                    by_tag[tag] = init_bucket()
                update_bucket(by_tag[tag], ranked_sources, expected_sources)

            if not hit5:
                misses.append({
                    "id": c.get("id"),
                    "question": q,
                    "expected": sorted(expected_sources),
                    "got_top5": ranked_sources[:5],
                    "best_score": best_score,
                    "tags": tags,
                })

    low_support_sources = sorted(
        [k for k, v in by_source.items() if v["total"] < 3]
    )

    results = {
        "overall": finalize_bucket(global_bucket),
        "by_source": {k: finalize_bucket(v) for k, v in sorted(by_source.items())},
        "by_tag": {k: finalize_bucket(v) for k, v in sorted(by_tag.items())},
        "mode_accuracy": (mode_correct / total_mode) if total_mode else 0.0,
        "total_cases": len(cases),
        "doc_cases": global_bucket["total"],
        "misses": misses,
        "warnings": {
            "low_doc_case_count": global_bucket["total"] < 30,
            "low_support_sources": low_support_sources,
        },
    }
    return results


def print_group_metrics(title, metrics):
    if not metrics:
        print(f"\n{title}: (none)")
        return

    print(f"\n{title}:")
    for name, m in metrics.items():
        print(
            f"- {name}: n={m['count']}, r@1={m['recall@1']:.3f}, r@3={m['recall@3']:.3f}, "
            f"r@5={m['recall@5']:.3f}, mrr={m['mrr']:.3f}"
        )


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

    overall = results["overall"]

    print("=== RAG Retrieval Eval ===")
    print(f"Total cases: {results['total_cases']}")
    print(f"Doc cases:   {results['doc_cases']}")
    print(f"Recall@1:    {overall['recall@1']:.3f}")
    print(f"Recall@3:    {overall['recall@3']:.3f}")
    print(f"Recall@5:    {overall['recall@5']:.3f}")
    print(f"MRR:         {overall['mrr']:.3f}")
    print(f"Mode acc:    {results['mode_accuracy']:.3f}")

    if results["warnings"]["low_doc_case_count"]:
        print("\n[WARN] doc_cases < 30: metrics are likely unstable.")
    if results["warnings"]["low_support_sources"]:
        print(
            "[WARN] low support per source (<3): "
            + ", ".join(results["warnings"]["low_support_sources"])
        )

    print_group_metrics("By source", results["by_source"])
    print_group_metrics("By tag", results["by_tag"])

    if results["misses"]:
        print("\n--- Misses (expected not in top5) ---")
        for m in results["misses"][:10]:
            print(f"- {m['id']}: {m['question']}")
            print(f"  expected: {m['expected']}")
            print(f"  got_top5: {m['got_top5']}")
            print(f"  best_score: {m['best_score']}")
            print(f"  tags: {m['tags']}")
    else:
        print("\nNo misses in top5.")


if __name__ == "__main__":
    main()
