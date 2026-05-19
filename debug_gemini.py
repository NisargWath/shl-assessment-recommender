import os, sys
from dotenv import load_dotenv
load_dotenv()

key   = os.getenv("GEMINI_API_KEY","")
model = os.getenv("GEMINI_MODEL","gemini-1.5-flash-latest")

print(f"KEY  : {'SET -> ' + key[:8]+'...' if key else 'MISSING - check .env'}")
print(f"MODEL: {model}")

import google.generativeai as genai
print(f"SDK  : {genai.__version__}")

genai.configure(api_key=key)

print("\nAvailable models with 'flash' in name:")
for m in genai.list_models():
    if "flash" in m.name:
        print(" ", m.name)

print(f"\nTrying model: {model}")
try:
    gm = genai.GenerativeModel(
        model_name=model,
        system_instruction="Reply only with valid JSON.",
    )
    chat = gm.start_chat(history=[])
    r = chat.send_message('Return {"reply":"ok","recommendations":[],"end_of_conversation":false}')
    print("SUCCESS:", r.text[:300])
except Exception as e:
    import traceback
    print("FAILED:")
    traceback.print_exc()