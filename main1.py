from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import requests
import time
import psutil
import os
import httpx

from indic_transliteration.sanscript import transliterate, DEVANAGARI, IAST, TAMIL

app = FastAPI(title="BhashaSetu Transliteration Service", version="1.2")

# ------------------ Enable CORS ------------------
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"  # optional for testing; remove in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Config ------------------
TRANSLATION_API_URL = "http://localhost:8000/translate"

# Map target languages to Sanscript schemes
lang_to_script = {
    "hin_Deva": DEVANAGARI,
    "tam_Taml": TAMIL,
    "IAST": IAST,
    # Add more as needed
}

# ------------------ Schemas ------------------
class TransliterationRequest(BaseModel):
    sentences: List[str]
    src_lang: str = "eng_Latn"
    tgt_lang: str = "hin_Deva"
    transliterate_langs: List[str] = ["IAST"]

class TransliterationResponse(BaseModel):
    original: str
    translated: str
    transliterations: Dict[str, str]
    translation_latency_ms: float
    transliteration_latency_ms: float
    translation_cost: float
    space_complexity_bytes: int
    time_complexity_ops: int

# ------------------ Routes ------------------
@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "Transliteration service is running."}

@app.post("/translate-and-transliterate", response_model=List[TransliterationResponse])
async def translate_and_transliterate(req: TransliterationRequest):
    print("[INFO] Received request for translation and transliteration.")

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    # --- Translation Phase ---
    translation_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(TRANSLATION_API_URL, json={
                "sentences": req.sentences,
                "src_lang": req.src_lang,
                "tgt_lang": req.tgt_lang
            })
        response.raise_for_status()
        translations = response.json().get("translations", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")
    translation_latency = (time.perf_counter() - translation_start) * 1000  # ms

    # Validate requested transliteration targets
    for lang in req.transliterate_langs:
        if lang not in lang_to_script:
            raise HTTPException(status_code=400, detail=f"Unsupported transliteration language/script: {lang}")

    # --- Transliteration Phase ---
    transliteration_start = time.perf_counter()
    result = []
    for original, translated in zip(req.sentences, translations):
        translit_outputs = {}
        for lang in req.transliterate_langs:
            try:
                translit_outputs[lang] = transliterate(
                    translated,
                    lang_to_script.get(req.tgt_lang, DEVANAGARI),
                    lang_to_script[lang]
                )
            except Exception as e:
                translit_outputs[lang] = f"[ERROR] {e}"

        transliteration_latency = (time.perf_counter() - transliteration_start) * 1000  # ms
        translation_cost = round(len(translated) * 0.00005, 6)
        mem_after = process.memory_info().rss
        space_complexity_bytes = mem_after - mem_before
        time_complexity_ops = len(original) * 5

        result.append(TransliterationResponse(
            original=original,
            translated=translated,
            transliterations=translit_outputs,
            translation_latency_ms=round(translation_latency, 3),
            transliteration_latency_ms=round(transliteration_latency, 3),
            translation_cost=translation_cost,
            space_complexity_bytes=space_complexity_bytes,
            time_complexity_ops=time_complexity_ops
        ))

    return result
