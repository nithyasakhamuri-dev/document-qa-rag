"""app.py - Flask web app for Document Q&A using RAG."""
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from rag import DocumentIndex, answer_question, chunk_text, load_text

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

index = None  # holds the current document's index (one document at a time)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global index
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith((".pdf", ".txt")):
        return jsonify(error="Please upload a .pdf or .txt file."), 400
    path = os.path.join(UPLOAD_DIR, secure_filename(file.filename))
    file.save(path)
    chunks = chunk_text(load_text(path))
    if not chunks:
        return jsonify(error="No readable text found in that file."), 400
    index = DocumentIndex(chunks)
    return jsonify(message=f"Loaded '{file.filename}' as {len(chunks)} chunks.")


@app.route("/ask", methods=["POST"])
def ask():
    if index is None:
        return jsonify(error="Upload a document first."), 400
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(error="Type a question first."), 400
    answer, passages = answer_question(index, question)
    return jsonify(
        answer=answer,
        sources=[{"text": t, "score": round(s, 3)} for t, s in passages],
    )


if __name__ == "__main__":
    app.run(debug=True)
