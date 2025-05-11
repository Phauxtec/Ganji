import os
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
    question = infer_marijuana_context(original)

    if DEBUG:
        print(f"[Input] 🗣 Original: {original}")
        print(f"[Input] 🔍 Inferred: {question}")
        print(f"[Input] 📘 Context: {context[:100]}...")

    if not question:
        return _empty_result()

    if not is_cannabis_related(question):
        return {
            "answer": "I'm here to help with cannabis-related questions! Feel free to ask about products, strains, effects, or recommendations.",
            "sentiment": "NEUTRAL",
            "confidence": 1.0,
            "escalate": False
        }

    if not context.strip():
        context = (
            "Cannabis is used for pain, anxiety, insomnia, appetite, and relaxation. "
            "Common products include flower, edibles, vape cartridges, tinctures, and topicals. "
            "Strains like indica are more sedative, while sativa is more energizing. "
            "Compounds like THC and CBD contribute to effects."
        )
        if DEBUG:
            print("[Context] ℹ️ Using fallback context.")

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

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Logit confidence margin
        start_conf = torch.topk(start_logits, 2).values
        end_conf = torch.topk(end_logits, 2).values
        logit_margin = (start_conf[0] - start_conf[1] + end_conf[0] - end_conf[1]).item()

        start = torch.argmax(start_logits).item()
        end = torch.argmax(end_logits).item() + 1

        if DEBUG:
            print(f"[QA] Logit margin: {logit_margin:.2f}")
            print(f"[QA] Start: {start}, End: {end}")

        if logit_margin < MIN_LOGIT_MARGIN:
            print("[QA] ❌ Low logit margin — rejecting")
        elif 0 <= start < end <= len(input_ids):
            tokens = input_ids[start:end]
            answer_text = tokenizer_qa.decode(tokens, skip_special_tokens=True).strip()
            print(f"[QA] ✅ Raw Answer: {answer_text}")

            # Filter out weak answers
            if len(answer_text.split()) < 2 or answer_text.lower() in {"yes", "no", "it is", "cannabis", "the answer is"}:
                print("[QA] ⚠️ Short or vague answer — rejecting")
                answer_text = ""
        else:
            print("[QA] ⚠️ Invalid span")
            answer_text = ""

    except Exception as e:
        print(f"[QA Error] ❌ {e}")
        answer_text = ""

    # === Sentiment Analysis
    sentiment = "NEUTRAL"
    confidence = 0.0
    try:
        sentiment_result = sentiment_pipeline(question)[0]
        sentiment = sentiment_result.get("label", "NEUTRAL")
        confidence = round(float(sentiment_result.get("score", 0.0)), 3)
        print(f"[Sentiment] 📊 {sentiment} ({confidence})")
    except Exception as e:
        print(f"[Sentiment Error] ❌ {e}")

    if not answer_text:
        answer_text = (
            "I'm not totally sure, but I can help guide you to the right cannabis product. "
            "Tell me a bit more about what you're looking for — effects, formats, or use case?"
        )

    escalate = sentiment == "NEGATIVE" and confidence >= CONFIDENCE_THRESHOLD

    return {
        "answer": answer_text,
        "sentiment": sentiment,
        "confidence": confidence,
        "escalate": escalate
    }

def _empty_result():
    return {
        "answer": "",
        "sentiment": "NEUTRAL",
        "confidence": 0.0,
        "escalate": False
    }
