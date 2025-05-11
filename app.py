from flask import Flask, request, jsonify
from mobilebert_handler import answer_question
from dotenv import load_dotenv
import os

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
        data = request.get_json()

        # Accept n8n or raw style
        question = data.get("message") or data.get("question")
        context = data.get("context", DEFAULT_CONTEXT)

        if not question:
            return jsonify({"error": "Missing 'message' or 'question'"}), 400

        result = answer_question(question, context)

        return jsonify({
            "question": question,
            "context": context,
            "answer": result["answer"],
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "escalate": result["escalate"]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
