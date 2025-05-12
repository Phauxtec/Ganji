from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from mobilebert_handler import (
    answer_question,
    get_embedding,
    summarize_text,
    classify_text,
    zero_shot_classify
)

load_dotenv()
app = Flask(__name__)

PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
DEFAULT_CONTEXT = os.getenv(
    "DEFAULT_CONTEXT",
    "You're a friendly, knowledgeable budtender who helps customers choose cannabis products based on their needs and preferences."
)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        mode = data.get("mode", "qa").lower()
        question = data.get("message") or data.get("question")
        context = data.get("context", DEFAULT_CONTEXT)

        if not question:
            return jsonify({"error": "Missing 'message' or 'question'"}), 400

        if mode == "qa":
            result = answer_question(question, context)
        elif mode == "embed":
            result = {"embedding": get_embedding(question)}
        elif mode == "summarize":
            result = {"summary": summarize_text(question)}
        elif mode == "classify":
            result = {"classification": classify_text(question)}
        elif mode == "zero-shot":
            labels = data.get("labels", ["product", "effect", "medical", "recreational"])
            result = zero_shot_classify(question, labels)
        else:
            return jsonify({"error": f"Unknown mode '{mode}'"}), 400

        return jsonify(result), 200

    except Exception as e:
        if DEBUG:
            print(f"[API Error] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    print(f"🚀 Serving NLP API on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
