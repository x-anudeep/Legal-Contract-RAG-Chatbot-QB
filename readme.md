# Legal Contract RAG Chatbot

## Project Description

This project is a chatbot that can answer questions about legal contracts using Retrieval-Augmented Generation (RAG).

The chatbot uses the CUAD (Contract Understanding Atticus Dataset), which contains real commercial contracts and labeled contract clauses. Instead of relying only on the language model, the system searches the contract for relevant information and uses that information to generate answers.

For example, a user can ask:

* What are the termination conditions?
* Is there a confidentiality clause?
* What happens if a party breaches the agreement?

The system retrieves the most relevant sections of the contract and provides an answer based on those sections.

## Dataset

CUAD (Contract Understanding Atticus Dataset)

https://zenodo.org/records/4595826

## Current Plan

* Load and preprocess contract data
* Split contracts into smaller chunks
* Generate embeddings for each chunk
* Store embeddings in a vector database
* Retrieve relevant chunks based on user questions
* Generate answers using an LLM

## Goal

The goal of this project is to build and evaluate a RAG-based chatbot for legal contract analysis and compare different retrieval approaches to improve answer accuracy.

## Person B Implementation

The app currently runs in mock mode, so it does not require Qdrant, BM25, CUAD data, or an LLM key. When Person A's artifacts are ready, switch `RETRIEVER_MODE=real` and point the backend at the Qdrant/BM25 handoff.

## Backend

```bash
python -m pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Smoke tests:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the governing law?","clause_filter":"Governing Law"}'
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the governing law?","clause_filter":"Governing Law","top_k":5}'
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed Vite URL, usually `http://localhost:5173`.

## Integration Switches

Default mock settings:

```bash
export RETRIEVER_MODE=mock
export RERANKER_MODE=simple
export GENERATOR_MODE=extractive
```

Gemini generation:

```bash
cp backend/.env.example backend/.env
```

Then edit `backend/.env` and set:

```bash
GENERATOR_MODE=gemini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash
```

Real retrieval later:

```bash
export RETRIEVER_MODE=real
export QDRANT_URL=http://localhost:6333
export QDRANT_COLLECTION=cuad_contracts
export BM25_INDEX_PATH=ingestion/bm25_index.pkl
```

OpenAI generation later:

```bash
export GENERATOR_MODE=openai
export OPENAI_API_KEY=your-key-here
```
