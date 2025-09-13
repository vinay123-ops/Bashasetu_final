import time
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

app = FastAPI()

# ✅ Enable CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"  # Optional for testing; remove in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Config ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
BATCH_SIZE = 6

model = None
tokenizer = None
ip = None

# ------------------ Schemas ------------------
class TranslationRequest(BaseModel):
    sentences: List[str]
    src_lang: str
    tgt_lang: str

class TranslationResponse(BaseModel):
    translations: List[str]
    latency_ms: float
    total_batches: int
    total_sentences: int

# ------------------ Load Model Once ------------------
@app.on_event("startup")
def load_model():
    global model, tokenizer, ip

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # ✅ Use float32 and no FlashAttention for CPU compatibility
    model_ = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(DEVICE)

    try:
        # torch.compile speeds things up if supported (PyTorch ≥2.0)
        model_ = torch.compile(model_)
    except Exception as e:
        print(f"torch.compile failed: {e}")

    model = model_
    ip = IndicProcessor(inference=True)
    print("Model, tokenizer, and processor loaded!")

# ------------------ Helper ------------------
def chunkify(lst, size):
    return [lst[i:i+size] for i in range(0, len(lst), size)]

# ------------------ Route ------------------
@app.post("/translate", response_model=TranslationResponse)
def translate_text(request_data: TranslationRequest):
    input_sentences = request_data.sentences
    src_lang = request_data.src_lang
    tgt_lang = request_data.tgt_lang

    if not input_sentences:
        raise HTTPException(status_code=400, detail="Empty sentence list")

    start_time = time.time()
    all_translations = []

    batches = chunkify(input_sentences, BATCH_SIZE)
    for batch in batches:
        preprocessed = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(
            preprocessed,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # ✅ Remove CUDA autocast and use_cache to prevent past_key_values bug
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                use_cache=False,
                min_length=0,
                max_length=64,
                num_beams=1,
                num_return_sequences=1,
            )

        decoded = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        translations = ip.postprocess_batch(decoded, lang=tgt_lang)
        all_translations.extend(translations)

        del inputs, outputs, decoded, translations
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    total_time = round((time.time() - start_time) * 1000, 2)

    return TranslationResponse(
        translations=all_translations,
        latency_ms=total_time,
        total_batches=len(batches),
        total_sentences=len(input_sentences),
    )
