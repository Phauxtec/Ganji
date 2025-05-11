import os
import random
import torch
import re

from flask import Flask, request, jsonify

from cannabis_nlp import is_cannabis_related, infer_marijuana_context

from transformers import (
    MobileBertTokenizer,
    MobileBertForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# === Environment ===
MODEL_NAME_QA = os.getenv("MODEL_NAME", "deepset/minilm-uncased-squad2")
MODEL_NAME_SENTIMENT = os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 512))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
MIN_LOGIT_MARGIN = float(os.getenv("MIN_LOGIT_MARGIN", 2.0))  # for QA confidence

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

# === Main inference function
def answer_question(question: str, context: str = "") -> dict:
    original = question.strip()
    inferred = infer_marijuana_context(original)
    cannabis_match = is_cannabis_related(inferred)

    print("\n======================= 🔍 Inference Trace =======================")
    print(f"[Input]         🧾 Original: {original}")
    print(f"[Input]         🧠 Inferred: {inferred}")
    print(f"[Filter]        🌿 Cannabis Related: {cannabis_match}")
    
    if not context.strip():
        context = (
            "Cannabis is used for pain, anxiety, insomnia, appetite, and relaxation. "
            "Common products include flower, edibles, vape cartridges, tinctures, and topicals. "
            "Strains like indica are more sedative, while sativa is more energizing. "
            "Compounds like THC and CBD contribute to effects."
        )
        print(f"[Context]       📘 Using fallback context.")

    else:
        print(f"[Context]       📘 Custom context detected ({len(context)} chars)")

    if not cannabis_match:
        print(f"[Decision]      🚫 Rejected as non-cannabis question. Sending FAQ help.\n")
        return {
	    "answer": (
	        "I'm here to help with cannabis-related questions. "
	        "Here are a few things I can help with:\n\n"
	        "• Which strains are good for sleep or pain?\n"
	        "• What’s the difference between sativa and indica?\n"
	        "• How do edibles or tinctures work?\n"
	        "• What’s a good beginner product?\n"
	        "• What does 'full spectrum' mean?\n\n"
	        "Feel free to ask me one of these or tell me what you're looking for! 🌿"
	    ),
	    "sentiment": "NEUTRAL",
	    "confidence": 1.0,
	    "escalate": False
	}

    # === QA Inference ===
    answer_text = ""
    try:
        inputs = tokenizer_qa.encode_plus(
            inferred,
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
        start_conf = outputs.start_logits.max().item()
        end_conf = outputs.end_logits.max().item()

        print(f"[QA]            📍 Start={start}, End={end}, Conf=[{start_conf:.2f}, {end_conf:.2f}]")

        if 0 <= start < end <= len(input_ids):
            tokens = input_ids[start:end]
            answer_text = tokenizer_qa.decode(tokens, skip_special_tokens=True).strip()
            print(f"[QA]            ✅ Answer: {answer_text}")
        else:
            print("[QA]            ⚠️ Invalid token span for answer.")

    except Exception as e:
        print(f"[QA Error]      ❌ {e}")
        answer_text = ""

    # === Sentiment ===
    sentiment = "NEUTRAL"
    confidence = 0.0
    try:
        sentiment_result = sentiment_pipeline(inferred)[0]
        sentiment = sentiment_result.get("label", "NEUTRAL")
        confidence = round(float(sentiment_result.get("score", 0.0)), 3)
        print(f"[Sentiment]     📊 {sentiment} ({confidence})")
    except Exception as e:
        print(f"[SentimentError] ❌ {e}")

    escalate = sentiment == "NEGATIVE" and confidence >= CONFIDENCE_THRESHOLD and bool(answer_text)

    print(f"[Decision]      🚦 Escalate: {escalate}")
    print("================================================================\n")

    return {
        "answer": answer_text or "I'm not totally sure, but I can help guide you. Can you clarify your question?",
        "sentiment": sentiment,
        "confidence": confidence,
        "escalate": escalate
    }




FAQ_SUGGESTIONS = [
    "Which strains are best for relaxation or sleep?",
    "What’s the difference between sativa and indica?",
    "How strong are edibles and how long do they last?",
    "Can cannabis help with anxiety or focus?",
    "What’s a good product for first-time users?"
]

def get_faq_suggestion():
    return "Here’s something you can ask:\n• " + "\n• ".join(random.sample(FAQ_SUGGESTIONS, 3))


def _empty_result():
    return {
        "answer": "",
        "sentiment": "NEUTRAL",
        "confidence": 0.0,
        "escalate": False
    }
