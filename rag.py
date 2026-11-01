"""rag.py - the RAG pipeline: load -> chunk -> vectorize -> retrieve -> answer.

Pure Python (no NumPy / scikit-learn), so it runs even where Windows Smart App
Control blocks compiled libraries.
"""
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter

from pypdf import PdfReader

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = "gemini-3.5-flash-lite"
MODELS = [GEMINI_MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL != GEMINI_MODEL else [])

STOP_WORDS = set(
    "a an and are as at be but by for from has have he her his how i if in into is it "
    "its of on or our she so than that the their them then there these they this to "
    "was we were what when where which who why will with you your do does did can".split()
)


# 1. LOAD -----------------------------------------------------------------
def load_text(path):
    """Read all text from a PDF or TXT file."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# 2. CHUNK ----------------------------------------------------------------
def chunk_text(text, chunk_size=80, overlap=20):
    """Split text into overlapping passages of about `chunk_size` words."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


# 3 + 4. VECTORIZE AND STORE ----------------------------------------------
def tokenize(text):
    """Lowercase, keep word characters, drop stop words."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS]


def normalize(vec):
    """Scale a {word: weight} vector to length 1 (so a dot product = cosine similarity)."""
    length = math.sqrt(sum(v * v for v in vec.values()))
    return {w: v / length for w, v in vec.items()} if length else vec


class DocumentIndex:
    """Holds the chunks and their TF-IDF vectors (our simple 'vector store')."""

    def __init__(self, chunks):
        self.chunks = chunks
        token_lists = [tokenize(c) for c in chunks]
        n = len(chunks)
        # document frequency: in how many chunks does each word appear?
        df = Counter(w for tokens in token_lists for w in set(tokens))
        # inverse document frequency: rare words get a higher weight
        self.idf = {w: math.log((1 + n) / (1 + d)) + 1 for w, d in df.items()}
        # TF-IDF vector for every chunk
        self.vectors = [
            normalize({w: tf * self.idf[w] for w, tf in Counter(tokens).items()})
            for tokens in token_lists
        ]

    # 5. RETRIEVE ---------------------------------------------------------
    def search(self, question, top_k=3):
        """Return the top_k chunks most similar to the question (cosine similarity)."""
        q_tf = Counter(w for w in tokenize(question) if w in self.idf)
        q_vec = normalize({w: tf * self.idf[w] for w, tf in q_tf.items()})
        scores = [
            sum(weight * vec.get(w, 0.0) for w, weight in q_vec.items())
            for vec in self.vectors
        ]
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], scores[i]) for i in best if scores[i] > 0]


# 6. ANSWER ---------------------------------------------------------------
def build_prompt(question, passages):
    context = "\n\n".join(f"[{i + 1}] {text}" for i, (text, _) in enumerate(passages))
    return (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. If the answer is not in the context, say you could not find it "
        "in the document.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def ask_llm(prompt):
    """Call the Gemini API. Returns None if no API key is set."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    last_error = None
    # Try the main model, then a lighter fallback. Retry when Google is busy.
    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for attempt in range(3):
            req = urllib.request.Request(
                url,
                data=body,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.load(resp)
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as err:
                last_error = err
                if err.code in (429, 500, 503):  # busy: wait and retry
                    time.sleep(2 * (attempt + 1))
                    continue
                break  # other errors (e.g. model not found): try the next model
    raise last_error


def answer_question(index, question):
    """Full query path: retrieve chunks, build the prompt, ask the LLM."""
    passages = index.search(question)
    if not passages:
        return "I could not find anything related to that in the document.", []
    try:
        answer = ask_llm(build_prompt(question, passages))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="ignore")[:300]
        return f"LLM request failed ({err.code}): {detail}", passages
    except (urllib.error.URLError, KeyError, ValueError) as err:
        return f"LLM request failed: {err}", passages
    if answer is None:
        answer = ("(No GEMINI_API_KEY set, so showing the best matching passages "
                  "instead of a generated answer.)")
    return answer, passages
