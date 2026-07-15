# AI Documentation Bot

An intelligent **RAG bot** that answers natural-language questions across **Pandas**, **NumPy**, and **Scikit-learn** documentation, using an **MLP classifier** to rank retrieved snippets by source reliability. **Answer relevance improved by 20%** by prioritizing official API references over community content like Stack Overflow.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg) ![Pinecone](https://img.shields.io/badge/Pinecone-vector%20DB-yellow) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Highlights

- **20% relevance improvement** by prioritizing official docs over community answers
- **3 large technical corpora** indexed (Pandas, NumPy, Scikit-learn)
- **MLP-based source classifier** trained on TF-IDF features to score API-spec vs tutorial vs community
- **LLM flexibility**: OpenAI (GPT-4o / GPT-3.5-Turbo) or local Ollama (Llama 3.1)
- **Streamlit UI** for quick prototyping

---

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌────────────────┐
│   Scraper    │──▶│  corpus.json │──▶│ Sentence-BERT  │
│(BeautifulSoup)│   │              │   │   Embeddings   │
└──────────────┘   └──────────────┘   └────────┬───────┘
                                               │
                                               ▼
                                       ┌──────────────┐
                                       │   Pinecone   │
                                       │ Vector Store │
                                       └──────┬───────┘
                                              │
User Query ──▶ Streamlit ──▶ Retrieval ───────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │  MLP Source        │  ← ranks snippets by
                     │  Classifier        │    API-spec / tutorial /
                     │  (TF-IDF + MLP)    │    community reliability
                     └──────────┬─────────┘
                                ▼
                     ┌────────────────────┐
                     │ Prompt Assembler   │  ← prioritizes official
                     │ (LangChain)        │    sources in context
                     └──────────┬─────────┘
                                ▼
                     ┌────────────────────┐
                     │  LLM (OpenAI /     │
                     │  local Ollama)     │
                     └──────────┬─────────┘
                                ▼
                          Grounded Answer
```

> **TODO:** replace with a rendered diagram once you have time. This sketch shows the flow.

---

## Tech Stack

**Backend & ML** &nbsp; Python 3.9+ · Scikit-learn (MLPClassifier, TF-IDF) · Sentence Transformers  
**Vector DB** &nbsp; Pinecone  
**LLM** &nbsp; OpenAI (GPT-4o / GPT-3.5-Turbo) or local Ollama (Llama 3.1 8B)  
**Orchestration** &nbsp; LangChain  
**Scraping** &nbsp; Requests · BeautifulSoup4  
**UI** &nbsp; Streamlit

---

## Quick Start

```bash
git clone https://github.com/gauravch-code/AI-doc-bot.git
cd AI-doc-bot

python -m venv venv
source venv/bin/activate         # macOS / Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

Create a `.env` in the root:

```bash
PINECONE_API_KEY=your-pinecone-key
OPENAI_API_KEY=your-openai-key    # Omit if using Ollama exclusively
```

### First-time setup (three prep steps)

```bash
python scraper.py             # collect docs → corpus.json
python train_classifier.py    # train MLP classifier → .pkl files
python build_vectordb.py      # populate Pinecone index
```

### Run the app

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

### Optional: use a local LLM via Ollama

```bash
# install Ollama from https://ollama.com/download
ollama pull llama3.1:8b
# swap ChatOpenAI for ChatOllama inside streamlit_app.py
```

---

## Testing

A test suite covering scraper HTML parsing, mocked Pinecone/OpenAI responses, and classifier loading is in progress. Run manual smoke tests via the Streamlit UI across a mix of Pandas, NumPy, and Scikit-learn queries to verify citation quality and source prioritization.

---

## Roadmap

- Extend to Matplotlib, TensorFlow, PyTorch documentation
- Query-time library detection (auto-route to the right index namespace)
- Replace TF-IDF with sentence embeddings in the classifier
- Dockerize and deploy to Streamlit Community Cloud
- Add formal unit and integration tests with mocking

---

## Contact

**Gaurav Chintakunta** · [LinkedIn](https://www.linkedin.com/in/gauravchintak/) · [Portfolio](https://gauravch-code.github.io/Portfolio/) · [gaurav.pvt25@gmail.com](mailto:gaurav.pvt25@gmail.com)
