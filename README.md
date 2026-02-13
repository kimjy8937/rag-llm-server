# rag-llm-server

FastAPI-based RAG server that indexes local documents, retrieves with FAISS + reranking, and answers using a Hugging Face-hosted LLM via the OpenAI SDK.

**Key Features**
- Document ingestion for `.txt`, `.md`, `.pdf`
- Chunking + embedding with SentenceTransformers
- FAISS vector store with on-disk persistence
- Cross-encoder reranking for better top-k results
- Simple chat session memory with rolling summary
- Retrieval eval script with recall@k and MRR

## Architecture Overview

1) **Ingestion**
- Reads files from `docs/` at startup (`.txt`, `.md`, `.pdf`).
- Splits text into ~500-character chunks.

2) **Embeddings**
- SentenceTransformers model: `all-MiniLM-L6-v2`.

3) **Vector Store**
- FAISS `IndexFlatL2` stored in `index/faiss.index`.
- Metadata stored in `index/docs.pkl`.

4) **Reranking**
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

5) **LLM**
- Hugging Face router via OpenAI SDK.
- Model: `moonshotai/Kimi-K2-Instruct-0905`.

## Project Structure

```
app/
  api/routes.py            # FastAPI endpoints
  config.py                # HF token + model name
  embeddings/embedder.py   # SentenceTransformers
  ingestion/document_loader.py
  llm/hf_llm.py            # OpenAI SDK wrapper
  rag/pipeline.py          # RAG orchestration + memory
  reranker/reranker.py
  vectorstore/faiss_store.py
docs/                      # Source documents to index
data/uploads/              # Uploaded files
eval/                      # Retrieval evaluation
index/                     # FAISS index + metadata
index_eval/                # Eval index output
```

## Setup

1) Create a virtual environment and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Set environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
```

## Run the Server

```bash
uvicorn app.main:app --reload
```

On startup, the server reads `docs/`, builds embeddings, and creates the FAISS index in `index/`.

## API Endpoints

**POST `/ask`**  
Ask a question with a session ID. The server decides whether to use documents based on reranker confidence.

Request:
```json
{
  "question": "What is the refund policy?",
  "session_id": "session1"
}
```

Response:
```json
{
  "answer": "....",
  "sources": [
    {"file": "05_refund_policy.md", "snippet": "...."}
  ],
  "mode": "document"
}
```

**POST `/documents`**  
Upload a document, chunk it, embed it, and add to the index.

```bash
curl -F "file=@/path/to/file.pdf" http://127.0.0.1:8000/documents
```

**POST `/reindex`**  
Rebuild the index from the `docs/` folder.

```bash
curl -X POST http://127.0.0.1:8000/reindex
```

## Evaluation

The eval script measures retrieval recall@k and MRR using `eval/eval_cases.jsonl`.

Run from repo root:
```bash
PYTHONPATH=. python eval/run_eval.py
```

Optional args:
```bash
python eval/run_eval.py --docs_dir docs --cases eval/eval_cases.jsonl --index_dir index_eval --candidate_k 20 --rerank_k 5 --threshold 0.3
```

## Notes

- `docs/` is the primary corpus used on startup.
- `data/uploads/` holds files uploaded through the API.
- FAISS index files live in `index/` and are overwritten when reindexing.
