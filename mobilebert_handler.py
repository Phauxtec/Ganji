# === Environment ===
import os
MODEL_NAME_QA = os.getenv("MODEL_NAME", "deepset/minilm-uncased-squad2")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
MODEL_NAME_SENTIMENT = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
NLI_MODEL = os.getenv("NLI_MODEL")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 512))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
MIN_LOGIT_MARGIN = float(os.getenv("MIN_LOGIT_MARGIN", 2.0))  # for QA confidence

import random
import re

# === Device setup ===
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from flask import Flask, request, jsonify

from cannabis_nlp import is_cannabis_related, infer_marijuana_context

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModel,
    pipeline
)

# === Load QA model
if MODEL_NAME_QA:
    try:
        tokenizer_qa = AutoTokenizer.from_pretrained(MODEL_NAME_QA)
        model_qa = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME_QA).to(DEVICE)
    except Exception as e:
        raise RuntimeError(f"[QA Model Error] Failed to load '{MODEL_NAME_QA}': {e}")

# === Sentiment ===
if SENTIMENT_MODEL:
    try:
        tokenizer_sent = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        model_sent = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL).to(DEVICE)
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_sent, tokenizer=tokenizer_sent,
                                      device=0 if DEVICE.type == "cuda" else -1)
    except Exception as e:
        raise RuntimeError(f"[Sentiment Model Error] Failed to load '{SENTIMENT_MODEL}': {e}")

# === Embeddings ===
if EMBEDDING_MODEL:
    try:
        tokenizer_embed = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model_embed = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)
    except Exception as e:
        raise RuntimeError(f"[Embedding Model Error] Failed to load '{EMBEDDING_MODEL}': {e}")

# === NLI / Zero-Shot ===
if NLI_MODEL:
    try:
        tokenizer_nli = AutoTokenizer.from_pretrained(NLI_MODEL)
        model_nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)
        nli_pipeline = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli,
                                device=0 if DEVICE.type == "cuda" else -1)
    except Exception as e:
        raise RuntimeError(f"[NLI Model Error] Failed to load '{NLI_MODEL}': {e}")

# === Summarizer ===
if SUMMARIZER_MODEL:
    try:
        summarizer_pipeline = pipeline("summarization", model=SUMMARIZER_MODEL,
                                       tokenizer=SUMMARIZER_MODEL,
                                       device=0 if DEVICE.type == "cuda" else -1)
    except Exception as e:
        raise RuntimeError(f"[Summarizer Error] Failed to load '{SUMMARIZER_MODEL}': {e}")

# === Classifier ===
if CLASSIFIER_MODEL:
    try:
        classifier_pipeline = pipeline("text-classification", model=CLASSIFIER_MODEL,
                                       tokenizer=CLASSIFIER_MODEL,
                                       device=0 if DEVICE.type == "cuda" else -1)
    except Exception as e:
        raise RuntimeError(f"[Classifier Error] Failed to load '{CLASSIFIER_MODEL}': {e}")


# === Primary QA Handler ===
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

    # === QA
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

        if 0 <= start < end <= len(input_ids):
            tokens = input_ids[start:end]
            answer_text = tokenizer_qa.decode(tokens, skip_special_tokens=True).strip()
            if DEBUG:
                print(f"[QA] ✅ Answer: {answer_text}")
        else:
            if DEBUG:
                print("[QA] ⚠️ Invalid span")
            answer_text = ""

    except Exception as e:
        if DEBUG:
            print(f"[QA Error] ❌ {e}")
        answer_text = ""

    # === Sentiment
    sentiment = "NEUTRAL"
    confidence = 0.0
    try:
        if SENTIMENT_MODEL:
            sentiment_result = sentiment_pipeline(question)[0]
            sentiment = sentiment_result.get("label", "NEUTRAL")
            confidence = round(float(sentiment_result.get("score", 0.0)), 3)
    except Exception as e:
        if DEBUG:
            print(f"[Sentiment Error] ❌ {e}")

    # === Escalation
    escalate = sentiment == "NEGATIVE" and confidence >= CONFIDENCE_THRESHOLD and answer_text != ""

    return {
        "question": original,
        "context": context,
        "answer": answer_text or "I'm not totally sure, but feel free to ask another way!",
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
# === Optional NLP Tasks ===

def get_embedding(text: str) -> list:
    if not EMBEDDING_MODEL or not model_embed:
        raise RuntimeError("Embedding model not loaded.")
    inputs = tokenizer_embed(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model_embed(**inputs)
    return outputs.last_hidden_state[:, 0].squeeze().cpu().tolist()


def summarize_text(text: str) -> str:
    if not SUMMARIZER_MODEL or not summarizer_pipeline:
        raise RuntimeError("Summarizer model not loaded.")
    result = summarizer_pipeline(text, max_length=120, min_length=30, do_sample=False)
    return result[0]["summary_text"]


def classify_text(text: str) -> dict:
    if not CLASSIFIER_MODEL or not classifier_pipeline:
        raise RuntimeError("Classifier model not loaded.")
    result = classifier_pipeline(text)[0]
    return {
        "label": result.get("label", "UNKNOWN"),
        "score": round(float(result.get("score", 0.0)), 3)
    }


def zero_shot_classify(text: str, candidate_labels: list) -> dict:
    if not NLI_MODEL or not nli_pipeline:
        raise RuntimeError("Zero-shot model not loaded.")
    result = nli_pipeline(text, candidate_labels)
    return {
        "label": result["labels"][0],
        "score": round(float(result["scores"][0]), 3),
        "all": dict(zip(result["labels"], map(float, result["scores"])))
    }
