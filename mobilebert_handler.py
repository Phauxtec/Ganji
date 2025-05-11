import os
import torch
import re

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

def infer_marijuana_context(message: str) -> str:
    text = message.lower().strip()

    if is_cannabis_related(text):
        return message

    # ---- Medical / Symptom use cases ----
    if any(word in text for word in ["pain", "cramps", "injury", "headache", "migraine", "arthritis", "back", "shoulder", "fibro", "ms", "nerve", "tension"]):
        return "What cannabis products help manage physical pain or inflammation?"

    if any(word in text for word in ["anxious", "stress", "nervous", "panic", "tight chest", "overthinking", "shaky", "edgy"]):
        return "What cannabis can I use to reduce anxiety or stress?"

    if any(word in text for word in ["can’t sleep", "insomnia", "wake up", "restless", "night owl", "trouble sleeping", "sleep aid"]):
        return "Which cannabis strains or edibles help with sleep and insomnia?"

    if any(word in text for word in ["adhd", "focus", "scatterbrained", "brain fog", "can’t think", "mental block", "stay on task", "clear headed"]):
        return "What cannabis helps with focus and mental clarity?"

    if any(word in text for word in ["depression", "sad", "mood", "no motivation", "empty", "feeling low", "mental health", "burnout"]):
        return "What cannabis products are good for mood uplift or depression?"

    if any(word in text for word in ["nausea", "vomit", "queasy", "chemo", "appetite loss", "can’t eat", "no hunger", "don’t feel like eating"]):
        return "What cannabis strains help with nausea or appetite stimulation?"

    if "ptsd" in text:
        return "What cannabis products are helpful for PTSD symptoms?"

    # ---- Effect-based needs ----
    if any(word in text for word in ["mellow", "relax", "chill", "calm", "unwind", "wind down", "zen", "decompress"]):
        return "What cannabis is best for relaxation or mellowing out?"

    if any(word in text for word in ["uplifted", "happy", "energized", "creative", "chatty", "sociable", "focus boost"]):
        return "What cannabis gives an energetic or creative high?"

    if any(word in text for word in ["couch lock", "heavy", "strong", "knockout", "lights out", "sleepy", "nighttime"]):
        return "Which cannabis is heavy hitting or sedative?"

    if any(word in text for word in ["functional", "low thc", "microdose", "day use", "no anxiety", "smooth", "clear high"]):
        return "What's a smooth, low-THC strain for functional daytime use?"

    # ---- Product formats & ambiguity ----
    if any(word in text for word in ["pen", "cart", "dab pen", "vape", "disposable", "battery", "that device"]):
        return "What is a cannabis vape pen or cart and how is it used?"

    if any(word in text for word in ["shatter", "badder", "crumble", "resin", "rosin", "live resin", "wax", "diamonds", "sauce", "extract", "terpy"]):
        return "Can you explain different types of cannabis concentrates?"

    if any(word in text for word in ["gummy", "brownie", "chocolate", "syrup", "lozenge", "mint", "edible", "drinkable"]):
        return "Can you recommend a cannabis edible or fast-acting gummy?"

    if any(word in text for word in ["tincture", "sublingual", "dropper", "under tongue"]):
        return "What is a cannabis tincture and when would I use it?"

    if any(word in text for word in ["topical", "cream", "rub", "transdermal", "patch", "balm", "salve", "gel"]):
        return "What are cannabis topicals and are they non-psychoactive?"

    # ---- Lifestyle / slang / vague requests ----
    if any(word in text for word in ["hook me up", "what you got", "need that", "zaza", "gas", "fire", "what's good", "plug", "some loud", "top shelf"]):
        return "What top-shelf or high-quality strains do you have available?"

    if any(word in text for word in ["hotbox", "session", "wake and bake", "nightcap", "blunt ride", "let’s sesh", "good for sharing"]):
        return "What’s a good cannabis product for a social or recreational session?"

    if any(word in text for word in ["fast", "quick", "no wait", "pickup now", "urgent", "in stock", "got it rn", "available"]) and "delivery" in text:
        return "Is delivery available now and what’s in stock?"

    if any(word in text for word in ["dose", "too much", "overdid", "how many mg", "what’s safe", "strong dose", "first time"]):
        return "What is a good cannabis dose for beginners or low tolerance users?"

    if any(word in text for word in ["recommend", "suggest", "what should i try", "got anything for", "what works for", "best option for"]):
        return "Can you suggest a cannabis product based on how I’m feeling?"

    if any(word in text for word in ["legal", "lab tested", "clean", "certificate", "coas", "licensed", "safe", "regulated"]):
        return "Are your cannabis products tested and compliant with regulations?"

    if any(word in text for word in ["sleep", "focus", "relax", "munchies", "giggles", "anxiety", "euphoria"]):
        return f"What cannabis products are good for {text.strip()}?"

    # ---- Emojis / cryptic slang triggers ----
    if any(symbol in text for symbol in ["🔥", "💨", "💤", "😴", "🧠", "🍃", "🍪", "😵", "😌", "🥴", "🌿"]):
        return "What cannabis product matches this kind of vibe or effect?"

    # ---- Ultimate fallback ----
    return f"What cannabis product or service would help if someone said: '{message}'"

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
