# SHL Conversational Assessment Recommendation System

A production-quality RAG-based conversational agent that recommends SHL
assessments using **FastAPI + Gemini 1.5 Flash + FAISS + sentence-transformers**.

---

## Architecture

```
User Request
    │
    ▼
POST /chat  (FastAPI)
    │
    ├─► Query Rewriting  (retriever.py → _rewrite_query)
    │       Short queries expanded with domain hints
    │
    ├─► FAISS Semantic Search  (retriever.py → retrieve_assessments)
    │       Top-20 candidates fetched, reranked by:
    │       cosine_similarity + 0.3 * keyword_overlap
    │       → Top-10 returned
    │
    ├─► RAG Prompt Builder  (llm.py → _build_context_block)
    │       Retrieved docs injected as <RETRIEVED_ASSESSMENTS> XML block
    │       Full conversation history included (max 8 turns)
    │
    ├─► Gemini 1.5 Flash  (llm.py → generate_response)
    │       Strict system prompt: catalog-only, JSON output,
    │       clarification logic, refusal logic, comparison handling
    │
    ├─► Hallucination Filter  (main.py)
    │       Any recommendation name NOT in retrieved docs is removed
    │
    └─► ChatResponse  (Pydantic validated)
            reply, recommendations[0-10], end_of_conversation
```

---

## Quick Start (Local)

### 1. Clone and set up environment

```bash
git clone <your-repo>
cd shl-assessment-system
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### 3. Ensure catalog files exist

If you've already scraped and built the index (you have `shl_catalog.csv`
and `shl_index.faiss`), skip to step 4.

Otherwise:
```bash
# Scrape the catalog
python scraper.py

# Build FAISS index
python embedding.py
```

### 4. Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open Swagger UI

Visit: http://localhost:8000/docs

---

## API Reference

### GET /health

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

### POST /chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to hire a Java developer"}
    ]
  }'
```

Response:
```json
{
  "reply": "Here are SHL assessments suitable for a Java developer role...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/...",
      "test_type": "Technical"
    }
  ],
  "end_of_conversation": false
}
```

---

## Example Multi-Turn Conversations

### Scenario 1: Technical Hiring

```bash
# Turn 1 - Initial query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hiring a .NET developer"}]}'

# Turn 2 - Refinement
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hiring a .NET developer"},
      {"role": "assistant", "content": "...previous response..."},
      {"role": "user", "content": "Only show me the WCF and MVC ones"}
    ]
  }'
```

### Scenario 2: Clarification Flow

```bash
# Vague query → agent asks for clarification
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need an assessment"}]}'

# Follow-up with details → agent recommends
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need an assessment"},
      {"role": "assistant", "content": "Could you tell me more about the role?"},
      {"role": "user", "content": "Entry level cashier position, retail store"}
    ]
  }'
```

### Scenario 3: Comparison

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Compare Accounts Payable (New) vs Accounts Receivable (New)"
      }
    ]
  }'
```

### Scenario 4: Off-topic Refusal

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'

# Response:
# {
#   "reply": "I can only help with SHL assessment recommendations...",
#   "recommendations": [],
#   "end_of_conversation": false
# }
```

---

## Deployment to Render

### Option A: render.yaml (Blueprint)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Blueprint
3. Connect your repo
4. In the Render dashboard, set the **GEMINI_API_KEY** environment variable
5. Deploy

### Option B: Manual Web Service

1. Go to Render → New → Web Service
2. Connect your GitHub repo
3. Set:
   - **Build Command**: `pip install -r requirements.txt && python embedding.py`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable: `GEMINI_API_KEY = <your key>`
5. Deploy

### Notes on Render Free Tier

- The free tier spins down after 15 minutes of inactivity (cold start ~30s)
- The FAISS index file (`shl_index.faiss`) and CSV must be committed to the repo
  OR the build command must regenerate them (slower cold start)
- Recommend committing both `shl_catalog.csv` and `shl_index.faiss` to the repo
  and removing `python embedding.py` from the build command for faster deploys

---

## Project Structure

```
shl-assessment-system/
├── main.py              # FastAPI app, request/response schemas, endpoints
├── retriever.py         # FAISS semantic search, query rewriting, reranking
├── llm.py               # Gemini 1.5 Flash integration, prompt engineering
├── embedding.py         # Build FAISS index from catalog CSV (run once)
├── scraper.py           # SHL catalog scraper (run once to build catalog)
├── shl_catalog.csv      # Scraped catalog data (commit this)
├── shl_index.faiss      # FAISS index (commit this OR build in CI)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .env                 # Your actual secrets (DO NOT COMMIT)
├── render.yaml          # Render.com deployment blueprint
├── Procfile             # Heroku/Render process definition
└── README.md            # This file
```

---

## Design Decisions

### Why FAISS + sentence-transformers instead of a hosted vector DB?

Assignment requires self-contained deployment. FAISS runs in-process with
zero network overhead. For production scale (>100K docs), switch to
Pinecone/Weaviate/Chroma.

### Why Gemini 1.5 Flash?

1. Free tier available for development
2. 1M token context window handles long conversation histories
3. Fast (sub-5s for our use case)
4. Instruction-following quality sufficient for JSON output mode

### Why stateless conversation?

Avoids server-side session storage, enables horizontal scaling, simplifies
deployment. Full history is sent each turn (max 8 turns × 2 = 16 messages).

### Why structured JSON output from LLM?

Eliminates post-processing regex heuristics. The system prompt enforces the
schema; `_extract_json()` handles the rare markdown fence edge case.

### How is hallucination prevented?

Three-layer defence:
1. **Prompt layer**: System prompt explicitly states "only use retrieved catalog"
2. **Context injection**: Only retrieved docs appear in the prompt context
3. **Post-processing filter**: `main.py` removes any recommendation whose name
   doesn't appear in the retrieved docs set

---

## Scoring Optimisation Notes

### Improving Recall@10

- Query rewriting with `_DOMAIN_HINTS` expands sparse technical queries
- Over-fetching (k×2) + reranking with keyword overlap boosts precision
- Embedding model `all-MiniLM-L6-v2` balances speed and quality

### For higher scores:

1. **Richer catalog**: Scrape full detail page text, not just the truncated description
2. **Cross-encoder reranking**: Add a `cross-encoder/ms-marco-MiniLM-L-6-v2` as a
   second-stage reranker for much better precision
3. **Hybrid retrieval**: Combine FAISS (dense) with BM25 (sparse) for better coverage
4. **Larger embedding model**: `all-mpnet-base-v2` or `bge-large-en-v1.5` for better
   semantic understanding (slower but more accurate)
