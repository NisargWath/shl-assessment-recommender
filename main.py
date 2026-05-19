"""
SHL Conversational Assessment Recommendation System
main.py - FastAPI application entry point

Architecture:
  User → POST /chat → conversation_orchestrator()
       → query_rewriter()  (makes sparse queries richer)
       → retrieve_assessments()  (FAISS semantic search + rerank)
       → build_llm_prompt()  (inject retrieved context)
       → Gemini 1.5 Flash  (grounded response + structured output)
       → validate_response()  (schema enforcement)
       → return JSON

Design decisions:
- Stateless: full conversation history sent each turn (≤8 turns)
- RAG-only: Gemini ONLY sees retrieved catalog snippets, never raw training knowledge
- Structured output: Gemini returns JSON parsed directly into response schema
- Clarification: LLM decides when query is too vague before retrieving
- Hallucination prevention: strict system prompt + catalog-only context
"""

import logging
import os
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from llm import generate_response
from retriever import retrieve_assessments

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("shl_api")

# ──────────────────────────────────────────────
# FastAPI app init
# ──────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Conversational agent for recommending SHL assessments using RAG + Gemini.",
    version="1.0.0",
)

# Allow all origins for assignment/demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────
class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v.strip()


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages list must not be empty")
        if len(v) > 16:
            raise ValueError("messages list exceeds maximum of 16 entries (8 turns)")
        return v


class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationItem]
    end_of_conversation: bool


# ──────────────────────────────────────────────
# Global exception handler – schema never breaks
# ──────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "reply": "I encountered an internal error. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness probe – always returns 200 if server is up."""
    return {"status": "ok"}


# ──────────────────────────────────────────────
# Chat endpoint
# ──────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main conversational endpoint.

    Flow:
    1. Extract conversation history + latest user query.
    2. Detect if query is a refusal target (off-topic).
    3. Retrieve top-K SHL assessments from FAISS.
    4. Pass history + retrieved context to Gemini.
    5. Parse Gemini JSON output into ChatResponse.
    6. Validate all recommendations are grounded in retrieved data.
    """
    messages = request.messages
    latest_query = messages[-1].content

    logger.info(f"Incoming query [{len(messages)} msgs]: {latest_query[:120]}")

    # ── Step 1: Retrieve relevant assessments ─────────────────
    # Even if the query is vague we retrieve; the LLM decides
    # whether to ask for clarification or return results.
    try:
        retrieved_docs = retrieve_assessments(latest_query, k=10)
        logger.info(f"Retrieved {len(retrieved_docs)} docs for query.")
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        retrieved_docs = []

    # ── Step 2: Call LLM with RAG context ─────────────────────
    try:
        llm_output = generate_response(
            messages=[m.model_dump() for m in messages],
            retrieved_docs=retrieved_docs,
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        return ChatResponse(
            reply="I'm having trouble generating a response right now. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )

    # ── Step 3: Ground-check recommendations ──────────────────
    # Only keep recommendations whose name appears in retrieved docs
    retrieved_names = {doc["name"].strip().lower() for doc in retrieved_docs}
    safe_recommendations = []
    for rec in llm_output.get("recommendations", []):
        if rec.get("name", "").strip().lower() in retrieved_names:
            safe_recommendations.append(rec)
        else:
            logger.warning(
                f"Hallucinated recommendation removed: {rec.get('name')}"
            )

    # Cap at 10 as per requirements
    safe_recommendations = safe_recommendations[:10]

    reply_text = llm_output.get("reply", "Here are my recommendations.")
    end_flag = llm_output.get("end_of_conversation", False)

    logger.info(
        f"Response: {len(safe_recommendations)} recommendations, "
        f"end_of_conversation={end_flag}"
    )

    return ChatResponse(
        reply=reply_text,
        recommendations=[RecommendationItem(**r) for r in safe_recommendations],
        end_of_conversation=bool(end_flag),
    )
