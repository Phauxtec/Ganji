import os
import torch
import re

from cannabis_nlp import is_cannabis_related, infer_marijuana_context

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

# === Main inference function
def answer_question(question: str, context: str = "") -> dict:
    original = question.strip()
    question = infer_marijuana_context(original)  # 🔁 Auto-infer vague input

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

    # === Provide fallback default context
    if not context.strip():
        context = (
            "Cannabis is used for pain, anxiety, insomnia, appetite, and relaxation. "
            "Common products include flower, edibles, vape cartridges, tinctures, and topicals. "
            "Strains like indica are more sedative, while sativa is more energizing. "
            "Compounds like THC and CBD contribute to effects."
        )
        if DEBUG:
            print("[Context] ℹ️ Fallback default context used.")

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

        if DEBUG:
            print(f"[QA] Start: {start}, End: {end}")
            print(f"[QA] Start logit max: {outputs.start_logits.max().item():.3f}")
            print(f"[QA] End logit max: {outputs.end_logits.max().item():.3f}")

        if 0 <= start < end <= len(input_ids):
            tokens = input_ids[start:end]
            answer_text = tokenizer_qa.decode(tokens, skip_special_tokens=True).strip()
            print(f"[QA] ✅ Answer: {answer_text}")
        else:
            print("[QA] ⚠️ Invalid span")
            answer_text = ""

    except Exception as e:
        print(f"[QA Error] ❌ {e}")
        answer_text = ""

    # === Run Sentiment
    sentiment = "NEUTRAL"
    confidence = 0.0
    try:
        sentiment_result = sentiment_pipeline(question)[0]
        sentiment = sentiment_result.get("label", "NEUTRAL")
        confidence = round(float(sentiment_result.get("score", 0.0)), 3)
        print(f"[Sentiment] 📊 {sentiment} ({confidence})")
    except Exception as e:
        print(f"[Sentiment Error] ❌ {e}")

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json(force=True)
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request"}), 400

        user_message = data['message']
        print(f"[API] Received: {user_message}")

        result = answer_question(user_message)
        return jsonify(result), 200

    except Exception as e:
        print(f"[API Error] ❌ {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8800))
    print(f"🚀 BERT handler running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)
