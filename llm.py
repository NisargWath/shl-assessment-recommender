"""
llm.py – Gemini integration for google-generativeai SDK 0.3.x
"""

import json, logging, os, re
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("shl_llm")

_GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
_MODEL_NAME        = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
_TEMPERATURE       = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
_genai = None

def _get_genai():
    global _genai
    if _genai is not None:
        return _genai
    import google.generativeai as genai
    genai.configure(api_key=_GEMINI_API_KEY)
    _genai = genai
    return _genai

# System prompt injected as first history pair (SDK 0.3.x workaround)
_SYSTEM_PROMPT = """You are the SHL Assessment Recommendation Assistant.

RULES — follow exactly:
1. Only recommend assessments listed in <RETRIEVED_ASSESSMENTS>. Never invent any.
2. Your ENTIRE response must be a single raw JSON object. No text before or after it.
   No markdown fences. No explanation. Just the JSON object itself.
3. JSON schema (required keys):
   {"reply": "string", "recommendations": [{"name": "...", "url": "...", "test_type": "..."}], "end_of_conversation": false}
4. recommendations: 1-10 items when recommending; empty [] when clarifying or refusing.
5. Copy name/url/test_type exactly from the retrieved block — verbatim, no changes.
6. If the query is vague, set recommendations to [] and ask ONE clarifying question in reply.
7. If the query is unrelated to hiring/assessments, politely refuse, recommendations [].
8. end_of_conversation: true only when user says goodbye/done.

CRITICAL: Output ONLY the raw JSON. First character must be { and last must be }."""

_SYSTEM_ACK = '{"reply": "Understood. I will output only raw JSON following the schema.", "recommendations": [], "end_of_conversation": false}'

def _build_context_block(docs: list[dict]) -> str:
    if not docs:
        return "<RETRIEVED_ASSESSMENTS>\nNone.\n</RETRIEVED_ASSESSMENTS>"
    lines = ["<RETRIEVED_ASSESSMENTS>"]
    for i, d in enumerate(docs, 1):
        lines.append(
            f"[{i}] Name: {d.get('name','N/A')} | "
            f"URL: {d.get('url','N/A')} | "
            f"TestType: {d.get('test_type','N/A')} | "
            f"Desc: {str(d.get('description',''))[:200]}"
        )
    lines.append("</RETRIEVED_ASSESSMENTS>")
    return "\n".join(lines)

def _extract_json(text: str) -> dict | None:
    """
    Robust JSON extractor. Handles:
    - Clean JSON
    - JSON wrapped in markdown fences
    - JSON embedded in prose (model ignores instructions sometimes)
    - Escaped JSON string (model returns {"reply": "{\"key\":...}"})
    """
    text = text.strip()

    # 1. Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # 2. Try direct parse
    try:
        data = json.loads(text)
        # If the reply field itself is a JSON string, unwrap it
        if isinstance(data, dict) and isinstance(data.get("reply"), str):
            inner = data["reply"].strip()
            if inner.startswith("{"):
                try:
                    inner_data = json.loads(inner)
                    if "recommendations" in inner_data:
                        return inner_data
                except json.JSONDecodeError:
                    pass
        return data
    except json.JSONDecodeError:
        pass

    # 3. Find the outermost {...} block
    start = text.find("{")
    if start != -1:
        # Walk to find matching closing brace
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break

    logger.error(f"JSON extract failed. Raw: {text[:400]}")
    return None

def _validate(data: dict) -> dict:
    reply = str(data.get("reply", "")).strip() or "Here are my recommendations."
    recs = []
    for r in (data.get("recommendations") or []):
        if isinstance(r, dict) and r.get("name") and r.get("url"):
            recs.append({
                "name":      str(r["name"]).strip(),
                "url":       str(r["url"]).strip(),
                "test_type": str(r.get("test_type", "Unknown")).strip(),
            })
    return {
        "reply": reply,
        "recommendations": recs[:10],
        "end_of_conversation": bool(data.get("end_of_conversation", False)),
    }

_VAGUE = re.compile(
    r"^(test|assessment|assessments|help|something|anything|hire|hiring|evaluate)\s*\??$",
    re.IGNORECASE,
)

def generate_response(messages: list[dict], retrieved_docs: list[dict]) -> dict:
    latest = messages[-1]["content"] if messages else ""

    if _VAGUE.match(latest.strip()):
        return {
            "reply": "I'd love to help! Could you tell me the job title, seniority level, and key skills you want to assess?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    genai = _get_genai()
    context = _build_context_block(retrieved_docs)

    # Build history: system bootstrap + prior turns (all but last)
    history = [
        {"role": "user",  "parts": [_SYSTEM_PROMPT]},
        {"role": "model", "parts": [_SYSTEM_ACK]},
    ]
    for msg in messages[:-1][-12:]:
        history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]],
        })

    last_message = (
        f"{context}\n\n"
        f"User request: {latest}\n\n"
        "Output ONLY the raw JSON object. No other text."
    )

    try:
        model    = genai.GenerativeModel(model_name=_MODEL_NAME)
        chat     = model.start_chat(history=history)
        response = chat.send_message(
            last_message,
            generation_config={
                "temperature":       _TEMPERATURE,
                "max_output_tokens": _MAX_OUTPUT_TOKENS,
            },
        )
        raw = response.text
        logger.info(f"Gemini raw (200 chars): {raw[:200]}")
    except Exception as e:
        import traceback; traceback.print_exc()
        logger.error(f"Gemini error: {type(e).__name__}: {e}")
        return {"reply": f"LLM error: {type(e).__name__}: {str(e)[:200]}", "recommendations": [], "end_of_conversation": False}

    parsed = _extract_json(raw)
    if not parsed:
        return {"reply": raw[:500], "recommendations": [], "end_of_conversation": False}

    return _validate(parsed)