import os
import torch
from transformers import (
    MobileBertTokenizer,
    MobileBertForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# === Environment ===
MODEL_NAME_QA = os.getenv("MODEL_NAME", "csarron/mobilebert-uncased-squad-v2")
MODEL_NAME_SENTIMENT = os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 512))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# === Device setup ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load QA model
try:
    tokenizer_qa = MobileBertTokenizer.from_pretrained(MODEL_NAME_QA)
    model_qa = MobileBertForQuestionAnswering.from_pretrained(MODEL_NAME_QA).to(DEVICE)
except Exception as e:
    raise RuntimeError(f"[QA Model Error] Failed to load '{MODEL_NAME_QA}': {e}")

# === Load Sentiment model
try:
    tokenizer_sent = AutoTokenizer.from_pretrained(MODEL_NAME_SENTIMENT)
    model_sent = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_SENTIMENT).to(DEVICE)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_sent, tokenizer=tokenizer_sent, device=0 if DEVICE.type == "cuda" else -1)
except Exception as e:
    raise RuntimeError(f"[Sentiment Model Error] Failed to load '{MODEL_NAME_SENTIMENT}': {e}")

# === Off-topic filter
CANNABIS_KEYWORDS = set([
    # 🧬 Cannabinoids
    "thc", "cbd", "cbn", "cbg", "thcv", "thca", "cbda", "cbc", "delta8", "delta9", "h4cbd", "hhc", "hhcp", "phc",

    # 🧪 Terpenes
    "terpenes", "terpene", "terps", "myrcene", "limonene", "linalool", "caryophyllene", "pinene", "humulene", "ocimene", "terpinolene", "nerolidol", "bisabolol", "valencene",

    # 🌱 Strains / Cultivars
    "indica", "sativa", "hybrid", "strain", "cultivar", "pheno", "genotype", "chemotype", "kush", "haze", "diesel", "purple", "cookie", "gelato", "runtz", "zkittlez", "mimosa", "gmo", "og",

    # 🚬 Consumption Forms
    "flower", "bud", "nug", "trim", "shake", "pre-roll", "preroll", "joint", "blunt", "spliff", "bong", "pipe", "bowl", "rig", "dab", "cart", "cartridge", "vape", "disposable", "battery", "e-rig",

    # 🍬 Ingestibles
    "edible", "gummy", "lozenge", "tincture", "capsule", "softgel", "infused", "syrup", "beverage", "drinkable", "nano", "sublingual", "roa",

    # 🧪 Concentrates
    "live resin", "live rosin", "sauce", "diamonds", "badder", "budder", "crumble", "wax", "shatter", "distillate", "isolate", "hash", "bubble hash", "kief",

    # 🧠 Effects / Experience
    "high", "stoned", "uplifted", "euphoric", "couch lock", "mellow", "body buzz", "clear headed", "giggly", "focused", "introspective", "creative", "stimulating", "psychoactive",

    # 🧑‍⚕️ Medical Use
    "anxiety", "insomnia", "inflammation", "chronic pain", "arthritis", "ptsd", "epilepsy", "nausea", "depression", "glaucoma", "fibromyalgia", "parkinson", "chemotherapy",

    # 📦 Packaging / Retail
    "menu", "dispensary", "order", "delivery", "pickup", "inventory", "testing", "certificate", "coa", "compliance", "label", "batch", "harvest date", "packaged date",

    # 💬 Slang & Culture
    "weed", "zaza", "loud", "gas", "top shelf", "mid", "regs", "dope", "fire", "exotics", "flame", "stash", "plug", "trap", "420", "710", "lit", "baked", "blazed", "rolling", "hotbox", "rip", "session"
])

CANNABIS_PHRASES = [
    "get high", "feel relaxed", "recommend a strain", "strongest edible", "indica for sleep",
    "sativa for focus", "low thc high cbd", "terpene profile", "strain effects", "what’s the best gummy",
    "dab temp", "rosin press", "how to roll", "how to dab", "fast acting", "sublingual tincture",
    "how many mg", "med card", "what's in your pre rolls", "entourage effect", "what's the terpene", "clean lab tested",
    "i need a vape", "full spectrum", "broad spectrum", "nano emulsified", "legal limit", "cannabinoid content",
    "lab results", "first time discount", "which cart is strongest", "weed for pain", "weed for anxiety",
    "what’s the effect", "how long does it last", "can i drive after", "delivery available",
    "batch info", "what’s the thca percentage", "weed with limonene", "pain relief strain",
    "gassy strain", "sweet terp", "earthy profile", "fruity terp", "heavy hitter", "recommend a hybrid"
]

import re

def is_cannabis_related(text: str) -> bool:
    text = text.lower()
    words = set(re.findall(r"\b\w+\b", text))

    # Keyword match (tokens)
    matched_keywords = words & CANNABIS_KEYWORDS
    if matched_keywords:
        print(f"[CannabisCheck] ✅ Keyword match: {', '.join(matched_keywords)}")
        return True

    # Phrase match (string contains)
    for phrase in CANNABIS_PHRASES:
        if phrase in text:
            print(f"[CannabisCheck] ✅ Phrase match: '{phrase}'")
            return True

    print("[CannabisCheck] ❌ No cannabis-related match")
    return False

# === Main inference function
def answer_question(question: str, context: str = "") -> dict:
    if DEBUG:
        print(f"[Input] Question: {question}")
        print(f"[Input] Context: {context}")

    if not question or not question.strip():
        return _empty_result()

    if not is_cannabis_related(question):
        return {
            "answer": "I'm here to help with cannabis-related questions! Feel free to ask about products, strains, effects, or recommendations.",
            "sentiment": "NEUTRAL",
            "confidence": 1.0,
            "escalate": False
        }

    # === Run QA
    answer_text = ""
    try:
        inputs = tokenizer_qa.encode_plus(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            padding="max_length"
        ).to(DEVICE)

        input_ids = inputs["input_ids"][0]
        with torch.no_grad():
            outputs = model_qa(**inputs)

        start = torch.argmax(outputs.start_logits).item()
        end = torch.argmax(outputs.end_logits).item() + 1

        # Defensive bounds
        if 0 <= start < end <= len(input_ids):
            tokens = input_ids[start:end]
            answer_text = tokenizer_qa.decode(tokens, skip_special_tokens=True).strip()
        else:
            answer_text = ""

    except Exception as e:
        if DEBUG:
            print(f"[QA Error] {e}")
        answer_text = ""

    # === Run Sentiment
    sentiment = "NEUTRAL"
    confidence = 0.0
    try:
        sentiment_result = sentiment_pipeline(question)[0]
        sentiment = sentiment_result.get("label", "NEUTRAL")
        confidence = round(float(sentiment_result.get("score", 0.0)), 3)
    except Exception as e:
        if DEBUG:
            print(f"[Sentiment Error] {e}")
        sentiment = "NEUTRAL"
        confidence = 0.0

    # === Escalation
    escalate = sentiment == "NEGATIVE" and confidence >= CONFIDENCE_THRESHOLD and answer_text != ""

    return {
        "answer": answer_text,
        "sentiment": sentiment,
        "confidence": confidence,
        "escalate": escalate
    }

# === Return an empty structure
def _empty_result():
    return {
        "answer": "",
        "sentiment": "NEUTRAL",
        "confidence": 0.0,
        "escalate": False
    }
