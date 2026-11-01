# Document Q&A using RAG (Flask)

Upload a PDF or TXT file and ask questions about it. The app finds the most relevant
passages in the document and sends them to an LLM (Google Gemini), which writes the answer.

## Files
| File | Job |
|---|---|
| `rag.py` | The RAG pipeline: load, chunk, TF-IDF vectors, cosine-similarity retrieval, LLM call |
| `app.py` | Flask routes: `/` page, `/upload` file, `/ask` question |
| `templates/index.html` | The web page (upload box, question box, answer + source passages) |
| `sample.txt` | A small test document about RAG |

## How to run (Windows; macOS/Linux commands are in brackets)
1. Install Python 3.9+ and open a terminal **inside this folder**
   (VS Code: File > Open Folder, then Terminal > New Terminal).
2. Install the libraries (one time):
   `pip install -r requirements.txt`
3. Get a free API key at https://aistudio.google.com/app/apikey, then set it in the same terminal:
   - Command Prompt: `set GEMINI_API_KEY=your_key_here`
   - PowerShell: `$env:GEMINI_API_KEY="your_key_here"`
   - [macOS/Linux: `export GEMINI_API_KEY=your_key_here`]
4. Start the app: `python app.py`
5. Open http://127.0.0.1:5000 in your browser, upload `sample.txt`, and ask
   "What is RAG?" or "Why does chunk overlap help?"

Without an API key the app still works: it shows the best-matching passages instead of a generated answer.

If you get a "model not found" error, set another model name:
`set GEMINI_MODEL=gemini-2.5-flash-lite` (use `export` on macOS/Linux).

## How it works
1. **Load**: read text from the PDF/TXT (`load_text`)
2. **Chunk**: split into overlapping 80-word passages (`chunk_text`)
3. **Vectorize**: TF-IDF turns each chunk into a vector (`DocumentIndex`)
4. **Store**: the vectors are kept in memory in `DocumentIndex.matrix`
5. **Retrieve**: the question is vectorized the same way; cosine similarity picks the top 3 chunks (`search`)
6. **Answer**: the chunks plus the question go into a prompt, and Gemini answers using only that context (`ask_llm`)

## Ideas to extend
- Replace TF-IDF with an embedding model for meaning-based matching
- Keep several documents and a persistent vector store
- Show chat history
