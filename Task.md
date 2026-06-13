# CUAD RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for legal contract review, built on the
[Contract Understanding Atticus Dataset (CUAD) v1](https://zenodo.org/records/4595826).

Given a natural language question such as "Does this contract have a non-compete clause?" or "What is the governing law?", the system retrieves the relevant contract passage and returns a grounded, cited answer.

---

## Dataset - CUAD v1

Download from: https://zenodo.org/records/4595826

After extracting `CUAD_v1.zip`, the folder structure is:

```
CUAD_v1/
|-- CUAD_v1.json              # SQuAD-format QA file - 510 contracts, 41 clause categories
|-- CUAD_v1_README.txt        # Official dataset documentation
|-- master_clauses.csv        # 511 rows x 83 columns - one row per contract, one column pair per clause
|-- full_contract_pdf/        # 510 raw PDFs (reference only)
|-- full_contract_txt/        # 510 plain-text contracts (primary input for ingestion)
`-- label_group_xlsx/         # 28 Excel files grouping the 41 clause categories
```

### What CUAD_v1.json contains

The JSON follows SQuAD 2.0 format. Each contract is split into paragraphs. For each
paragraph, 41 questions are asked - one per clause category. Each question has:

- `context` - the raw paragraph text from the contract
- `qas[]` - list of 41 QA pairs, one per clause category
  - `id` - the clause category name (e.g. "Governing Law")
  - `question` - the natural language question for that category
  - `is_impossible` - false if the clause is present in this paragraph, true if absent
  - `answers[]` - list of {text, answer_start} spans when is_impossible is false

### What master_clauses.csv contains

- Row 1: headers - `Filename`, then one column pair per clause (clause text + derived answer)
- Rows 2-511: one contract per row
- For each clause category there are two columns:
  - `[Category Name]` - the full clause text extracted from the contract
  - `[Category Name] Answer` - the normalized answer (e.g. "Nevada", "Yes", "No", "5/8/2014")
- Empty cell = clause not present in that contract

### The 41 clause categories

33 categories have Yes/No answers. 8 categories return extracted values:

| Category | Answer type |
|---|---|
| Document Name | Contract name |
| Parties | Entity names |
| Agreement Date | Date (mm/dd/yyyy) |
| Effective Date | Date (mm/dd/yyyy) |
| Expiration Date | Date (mm/dd/yyyy) or Perpetual |
| Renewal Term | Number of years/months |
| Notice to Terminate Renewal | Number of days/months |
| Governing Law | State or country name |
| All remaining 33 categories | Yes / No |

---

## Repository Structure

```
cuad-rag-chatbot/
|-- README.md
|-- data/                         # place CUAD_v1/ contents here
|   |-- CUAD_v1.json
|   |-- master_clauses.csv
|   |-- full_contract_txt/
|   `-- label_group_xlsx/
|-- ingestion/                    # Person A - data pipeline
|   |-- extract_chunks.py
|   |-- embed_chunks.py
|   |-- load_qdrant.py
|   |-- build_bm25.py
|   `-- export_metadata.py
|-- backend/                      # Person B - RAG pipeline + API
|   |-- retriever.py
|   |-- reranker.py
|   |-- generator.py
|   |-- main.py                   # FastAPI app
|   `-- requirements.txt
|-- frontend/                     # Person B - React UI
|   |-- src/
|   |   |-- App.jsx
|   |   |-- ClauseSelector.jsx
|   |   |-- ChatInput.jsx
|   |   `-- AnswerCard.jsx
|   `-- package.json
`-- evaluation/                   # Person C - eval harness
    |-- build_test_set.py
    |-- retrieval_eval.py
    |-- generation_eval.py
    |-- ragas_eval.py
    `-- dashboard.py
```

---

## Person A - Data Ingestion Pipeline

**Owns:** `ingestion/`  
**Input files:** `data/CUAD_v1.json`, `data/full_contract_txt/`  
**Output files:** Running Qdrant collection + `bm25_index.pkl` + `chunks_metadata.csv`

Person A converts the raw CUAD dataset into a searchable vector index. The outputs are
handed to Person B (Qdrant + BM25) and Person C (metadata CSV).

---

### Step A-1 - Understand the JSON structure

Before writing any pipeline code, run this to confirm you can navigate the JSON correctly.

```python
import json

with open("data/CUAD_v1.json") as f:
    data = json.load(f)

print(f"Number of contracts: {len(data['data'])}")       # 510

contract = data['data'][0]
print(f"Contract title: {contract['title']}")
print(f"Number of paragraphs: {len(contract['paragraphs'])}")

para = contract['paragraphs'][0]
print(f"\nParagraph text (first 300 chars):\n{para['context'][:300]}")

qa = para['qas'][0]
print(f"\nClause category: {qa['id']}")
print(f"Question: {qa['question']}")
print(f"Clause present: {not qa['is_impossible']}")
print(f"Answer spans: {qa['answers']}")
```

Expected output: you should see a contract filename, a paragraph of legal text, a clause
category name like "Governing Law", and either answer spans or an empty list.

---

### Step A-2 - Extract chunks with metadata

Each paragraph in the JSON becomes one chunk. Cross-reference the `qas[]` array to attach
the clause category labels and answers as metadata. Clean redaction markers and
`<omitted>` tags that appear in the raw text.

`ingestion/extract_chunks.py`

```python
import json, re, os

def clean_text(text):
    """Remove CUAD-specific artifacts from contract text."""
    text = text.replace("<omitted>", " ... ")          # annotators used this for skipped text
    text = re.sub(r'\[?\*+\]?', '[REDACTED]', text)   # redacted sections appear as ***
    text = re.sub(r'_{3,}', '[REDACTED]', text)        # also appear as ___
    text = re.sub(r'\s+', ' ', text)                   # normalize whitespace from PDF conversion
    return text.strip()

def extract_chunks(json_path):
    with open(json_path) as f:
        data = json.load(f)

    chunks = []
    for contract in data['data']:
        contract_name = contract['title']

        for chunk_index, para in enumerate(contract['paragraphs']):
            text = clean_text(para['context'])

            # skip very short fragments - they are usually headers or page numbers
            if len(text) < 50:
                continue

            # collect clause categories present in this paragraph
            # is_impossible=False means the clause WAS found in this paragraph
            # is_impossible=True  means annotators checked and the clause is NOT here
            clause_categories = []
            answers = []
            for qa in para['qas']:
                if not qa['is_impossible'] and qa['answers']:
                    clause_categories.append(qa['id'])
                    answers.append(qa['answers'][0]['text'])

            chunks.append({
                "contract_name":     contract_name,
                "chunk_index":       chunk_index,
                "text":              text,
                "clause_categories": clause_categories,
                "answers":           answers,
            })

    print(f"Extracted {len(chunks)} chunks from {len(data['data'])} contracts")
    return chunks

if __name__ == "__main__":
    chunks = extract_chunks("data/CUAD_v1.json")
```

**What this produces:** A list of dicts, one per paragraph. Each dict has:
- `contract_name` - which of the 510 contracts this came from
- `chunk_index` - position of this paragraph within that contract
- `text` - the cleaned paragraph text (this is what gets embedded)
- `clause_categories` - list of clause types found in this paragraph (e.g. ["Governing Law"])
- `answers` - list of corresponding answers (e.g. ["Nevada"])

A paragraph can contain multiple clause types - e.g. a single paragraph may be labeled
as both `Termination for Convenience` and `Notice to Terminate Renewal`.

---

### Step A-3 - Embed all chunks

Use `BAAI/bge-large-en-v1.5` - it outperforms general-purpose models on legal text.
Set `normalize_embeddings=True` - required for cosine similarity search in Qdrant.

`ingestion/embed_chunks.py`

```python
import pickle
from sentence_transformers import SentenceTransformer
from extract_chunks import extract_chunks

def embed_chunks(chunks):
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    texts = [c['text'] for c in chunks]

    # batch_size=32 is safe on CPU and GPU
    # normalize_embeddings=True is required for cosine similarity
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()

    # save to disk - do not re-run embedding every time
    with open("ingestion/embeddings_cache.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} embeddings to embeddings_cache.pkl")
    return chunks

if __name__ == "__main__":
    chunks = extract_chunks("data/CUAD_v1.json")
    embed_chunks(chunks)
```

Embedding size for `bge-large-en-v1.5` is **1024 dimensions**. This must match the
vector size you configure in Qdrant in the next step.

---

### Step A-4 - Load into Qdrant

Start Qdrant locally using Docker before running this script.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

`ingestion/load_qdrant.py`

```python
import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

def load_into_qdrant(chunks, collection_name="cuad_contracts"):
    client = QdrantClient("localhost", port=6333)

    # recreate collection - safe to run multiple times
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,                  # must match bge-large-en-v1.5 output size
            distance=Distance.COSINE    # must match normalize_embeddings=True
        )
    )

    points = []
    for idx, chunk in enumerate(chunks):
        points.append(PointStruct(
            id=idx,
            vector=chunk['embedding'],
            payload={
                # everything except the embedding is stored as payload
                # Person B queries these fields for retrieval
                # Person C uses chunk_id + contract_name + clause_categories for eval
                "contract_name":     chunk['contract_name'],
                "chunk_index":       chunk['chunk_index'],
                "text":              chunk['text'],
                "clause_categories": chunk['clause_categories'],
                "answers":           chunk['answers'],
            }
        ))

    # upload in batches of 256
    batch_size = 256
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name, points[i:i+batch_size])
        print(f"Uploaded {min(i+batch_size, len(points))}/{len(points)}")

    count = client.get_collection(collection_name).vectors_count
    print(f"Qdrant collection '{collection_name}' loaded with {count} vectors")

if __name__ == "__main__":
    with open("ingestion/embeddings_cache.pkl", "rb") as f:
        chunks = pickle.load(f)
    load_into_qdrant(chunks)
```

---

### Step A-5 - Build BM25 index

Person B needs this for hybrid retrieval. BM25 is a keyword-based ranking function that
performs well on legal terminology (`indemnify`, `arbitration`, `termination`) that dense
embeddings sometimes miss.

`ingestion/build_bm25.py`

```python
import pickle, re
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def build_bm25(chunks):
    corpus = [tokenize(c['text']) for c in chunks]
    bm25   = BM25Okapi(corpus)

    # save the BM25 index and the original texts together
    # Person B loads this file directly
    with open("ingestion/bm25_index.pkl", "wb") as f:
        pickle.dump({
            "bm25":         bm25,
            "corpus_texts": [c['text'] for c in chunks],
            "chunk_ids":    list(range(len(chunks)))
        }, f)
    print(f"BM25 index saved - {len(chunks)} documents")

if __name__ == "__main__":
    with open("ingestion/embeddings_cache.pkl", "rb") as f:
        chunks = pickle.load(f)
    build_bm25(chunks)
```

---

### Step A-6 - Export metadata CSV for Person C

Person C's evaluation harness needs a flat file mapping every chunk to its contract,
clause category, and ground truth answer. This is separate from the Qdrant index.

`ingestion/export_metadata.py`

```python
import csv, pickle

def export_metadata(chunks, output_path="evaluation/chunks_metadata.csv"):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "chunk_id", "contract_name", "chunk_index",
            "clause_category", "answer", "text_preview"
        ])
        writer.writeheader()

        for idx, chunk in enumerate(chunks):
            if not chunk['clause_categories']:
                # paragraph with no labeled clauses - still export with empty category
                writer.writerow({
                    "chunk_id":       idx,
                    "contract_name":  chunk['contract_name'],
                    "chunk_index":    chunk['chunk_index'],
                    "clause_category": "",
                    "answer":         "",
                    "text_preview":   chunk['text'][:150]
                })
            else:
                # one row per clause category in this chunk
                for cat, ans in zip(chunk['clause_categories'], chunk['answers']):
                    writer.writerow({
                        "chunk_id":       idx,
                        "contract_name":  chunk['contract_name'],
                        "chunk_index":    chunk['chunk_index'],
                        "clause_category": cat,
                        "answer":         ans,
                        "text_preview":   chunk['text'][:150]
                    })

    print(f"Metadata exported to {output_path}")

if __name__ == "__main__":
    with open("ingestion/embeddings_cache.pkl", "rb") as f:
        chunks = pickle.load(f)
    export_metadata(chunks)
```

---

### Person A - Handoffs

After completing all steps above, share the following with your teammates:

| What | Where | Goes to |
|---|---|---|
| Qdrant running | `localhost:6333`, collection = `cuad_contracts` | Person B |
| BM25 index | `ingestion/bm25_index.pkl` | Person B |
| Metadata CSV | `evaluation/chunks_metadata.csv` | Person C |
| Vector dimension | `1024`, distance = `COSINE` | Person B |

---

---

## Person B - RAG Pipeline + API + Frontend

**Owns:** `backend/`, `frontend/`  
**Receives from A:** Qdrant at `localhost:6333`, `ingestion/bm25_index.pkl`  
**Output:** `POST /chat` endpoint + React UI

Person B owns everything the user interacts with - the retrieval logic, the LLM
generation, the FastAPI backend, and the React frontend.

**Important:** Do not wait for Person A to start. In the first phase, use a mock
retriever with 10 hardcoded chunks. Replace it with the real Qdrant retriever once
Person A's index is ready.

---

### Step B-1 - Set up with a mock retriever (run immediately, no dependency on A)

Create this mock before A's index is ready so you can build and test the full pipeline.

`backend/retriever.py` (mock section)

```python
# Use this in the first phase before Qdrant is ready
MOCK_CHUNKS = [
    {
        "text": "This Agreement shall be governed by the laws of the State of Nevada.",
        "contract_name": "SampleCo_License_2019",
        "clause_categories": ["Governing Law"],
        "answers": ["Nevada"]
    },
    {
        "text": "Either party may terminate this Agreement upon 30 days written notice.",
        "contract_name": "SampleCo_License_2019",
        "clause_categories": ["Termination for Convenience", "Notice to Terminate Renewal"],
        "answers": ["Yes", "30 days"]
    },
    {
        "text": "During the term and for two years thereafter, neither party shall solicit "
                "or hire employees of the other party.",
        "contract_name": "AcmeCorp_Service_2020",
        "clause_categories": ["No-Solicit of Employees"],
        "answers": ["Yes"]
    },
]

def mock_retrieve(query: str, clause_filter=None, top_k=5):
    if clause_filter:
        filtered = [c for c in MOCK_CHUNKS
                    if clause_filter in c['clause_categories']]
        return filtered[:top_k] if filtered else MOCK_CHUNKS[:top_k]
    return MOCK_CHUNKS[:top_k]
```

---

### Step B-2 - Build the hybrid retriever (swap in once A's Qdrant is ready)

Hybrid retrieval combines:
- **Dense search** - semantic similarity via Qdrant vector search
- **BM25** - keyword matching, critical for legal terms (`indemnify`, `escrow`, `arbitration`)
- **Reciprocal Rank Fusion (RRF)** - merges both ranked lists without needing to tune weights

`backend/retriever.py`

```python
import pickle, re
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

client = QdrantClient("localhost", port=6333)
model  = SentenceTransformer("BAAI/bge-large-en-v1.5")

with open("ingestion/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

bm25         = bm25_data['bm25']
corpus_texts = bm25_data['corpus_texts']
chunk_ids    = bm25_data['chunk_ids']

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def reciprocal_rank_fusion(rank_lists, k=60):
    """Merge multiple ranked lists. Higher score = better rank."""
    scores = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def retrieve(query: str, clause_filter=None, top_k=20):
    # --- dense retrieval via Qdrant ---
    q_vec = model.encode(query, normalize_embeddings=True).tolist()

    qdrant_filter = None
    if clause_filter:
        # filter to only chunks that contain the specified clause category
        # clause_categories is stored as a list in the Qdrant payload
        qdrant_filter = Filter(must=[
            FieldCondition(key="clause_categories",
                           match=MatchValue(value=clause_filter))
        ])

    dense_hits = client.search(
        "cuad_contracts", q_vec,
        query_filter=qdrant_filter,
        limit=top_k
    )
    dense_ids = [h.id for h in dense_hits]

    # --- BM25 keyword retrieval ---
    tokens     = tokenize(query)
    bm25_scores = bm25.get_scores(tokens)
    bm25_ids   = sorted(range(len(bm25_scores)),
                        key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    # --- fuse both ranked lists ---
    fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids])[:top_k]

    # fetch full payloads from Qdrant
    results = client.retrieve("cuad_contracts", fused_ids, with_payload=True)
    return [r.payload for r in results]
```

---

### Step B-3 - Add cross-encoder reranking

The cross-encoder reads the query and each candidate passage together and scores their
relevance jointly. This is more accurate than embedding similarity alone and measurably
improves answer quality.

`backend/reranker.py`

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: list, top_k=5) -> list:
    """
    Takes the top-20 retrieved chunks and reranks them.
    Returns only the top_k most relevant.
    """
    pairs  = [(query, c['text']) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]
```

---

### Step B-4 - Build the LLM generation step

The system prompt forces the LLM to answer only from retrieved context and explicitly
handle the case where a clause is absent - important because 33 of the 41 CUAD
categories are Yes/No questions where the correct answer is sometimes "No, this clause
is not present."

`backend/generator.py`

```python
import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

SYSTEM_PROMPT = """You are a legal contract analyst assistant.

You will be given excerpts from commercial contracts and a question about a specific 
clause type. Your job is to answer the question strictly based on the provided excerpts.

Rules:
1. Answer ONLY from the provided contract text. Do not use outside knowledge.
2. If the relevant clause is not present in any of the excerpts, respond with:
   "This clause was not found in the retrieved contract sections."
3. Always cite which contract and clause type your answer is drawn from.
4. For Yes/No questions, lead with Yes or No, then quote the relevant text.
5. For date or entity extraction, provide the exact extracted value first, then context."""

def generate(query: str, chunks: list) -> dict:
    # build the context block from retrieved chunks
    context_blocks = []
    for i, c in enumerate(chunks):
        clause_label = ', '.join(c['clause_categories']) if c['clause_categories'] else 'Unlabeled'
        context_blocks.append(
            f"[Excerpt {i+1}]\n"
            f"Contract: {c['contract_name']}\n"
            f"Clause type: {clause_label}\n"
            f"Text: {c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Contract excerpts:\n\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0      # deterministic output - important for eval reproducibility
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "contract":         c['contract_name'],
                "clause_categories": c['clause_categories'],
                "text_preview":     c['text'][:300]
            }
            for c in chunks
        ]
    }
```

---

### Step B-5 - Wire everything into FastAPI

`backend/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import retrieve, mock_retrieve
from reranker import rerank
from generator import generate

app = FastAPI(title="CUAD RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question:      str
    clause_filter: str | None = None   # e.g. "Governing Law" - one of the 41 CUAD categories

class ChatResponse(BaseModel):
    answer:  str
    sources: list

USE_MOCK = False   # set True before Person A's Qdrant is ready

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Step 1: retrieve candidates
    if USE_MOCK:
        candidates = mock_retrieve(req.question, req.clause_filter, top_k=20)
    else:
        candidates = retrieve(req.question, req.clause_filter, top_k=20)

    # Step 2: rerank to top 5
    top_chunks = rerank(req.question, candidates, top_k=5)

    # Step 3: generate answer from top chunks
    result = generate(req.question, top_chunks)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}
```

```bash
# run the API
pip install fastapi uvicorn openai qdrant-client sentence-transformers rank-bm25
uvicorn backend.main:app --reload --port 8000
```

---

### Step B-6 - Build the React frontend

The UI has three core components: a clause category selector (the 41 CUAD categories as
a dropdown), a chat input, and an answer card with source attribution.

```bash
npx create-react-app frontend
cd frontend
npm install axios
```

`frontend/src/App.jsx`

```jsx
import { useState } from "react"
import axios from "axios"

const CLAUSE_CATEGORIES = [
  "Governing Law", "Non-Compete", "Termination for Convenience",
  "Expiration Date", "Renewal Term", "License Grant",
  "Anti-Assignment", "Change of Control", "Cap on Liability",
  "Uncapped Liability", "IP Ownership Assignment", "Audit Rights",
  // add all 41 CUAD categories
]

export default function App() {
  const [question,    setQuestion]    = useState("")
  const [clauseFilter, setClauseFilter] = useState("")
  const [answer,      setAnswer]      = useState(null)
  const [sources,     setSources]     = useState([])
  const [loading,     setLoading]     = useState(false)

  async function handleSubmit() {
    if (!question.trim()) return
    setLoading(true)
    try {
      const res = await axios.post("http://localhost:8000/chat", {
        question,
        clause_filter: clauseFilter || null
      })
      setAnswer(res.data.answer)
      setSources(res.data.sources)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 800, margin: "40px auto", padding: "0 20px" }}>
      <h1>CUAD Contract Chatbot</h1>

      {/* Clause category filter */}
      <select value={clauseFilter} onChange={e => setClauseFilter(e.target.value)}>
        <option value="">All clause types</option>
        {CLAUSE_CATEGORIES.map(c => (
          <option key={c} value={c}>{c}</option>
        ))}
      </select>

      {/* Question input */}
      <div style={{ display: "flex", gap: 8, margin: "12px 0" }}>
        <input
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSubmit()}
          placeholder="Ask a question about the contracts..."
          style={{ flex: 1, padding: "8px 12px" }}
        />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Searching..." : "Ask"}
        </button>
      </div>

      {/* Answer */}
      {answer && (
        <div style={{ background: "#f0fdf4", padding: 16, borderRadius: 8 }}>
          <strong>Answer</strong>
          <p style={{ marginTop: 8 }}>{answer}</p>
        </div>
      )}

      {/* Sources */}
      {sources.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <strong>Sources</strong>
          {sources.map((s, i) => (
            <div key={i} style={{ border: "1px solid #e2e8f0",
                                  borderRadius: 6, padding: 12, marginTop: 8 }}>
              <div><strong>Contract:</strong> {s.contract}</div>
              <div><strong>Clause:</strong> {s.clause_categories?.join(", ") || "-"}</div>
              <div style={{ marginTop: 6, color: "#555",
                            fontSize: 13 }}>{s.text_preview}...</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

---

### Person B - Handoffs

| What | Where | Goes to |
|---|---|---|
| `/chat` endpoint | `POST localhost:8000/chat` | Person C |
| Request schema | question (str), clause_filter (str or null) | Person C |
| Response schema | answer (str), sources (list of contract, clause_categories, text_preview) | Person C |

---

---

## Person C - Evaluation Harness

**Owns:** `evaluation/`  
**Receives from A:** `evaluation/chunks_metadata.csv`  
**Receives from B:** `POST localhost:8000/chat` API  
**Input files:** `data/master_clauses.csv`, `data/label_group_xlsx/`

Person C owns all measurement. The first two steps have no dependency on A or B and
can be completed immediately using only `master_clauses.csv`.

---

### Step C-1 - Explore master_clauses.csv (start immediately, no dependencies)

```python
import pandas as pd

df = pd.read_csv("data/master_clauses.csv")
print(f"Shape: {df.shape}")          # (511, 83) - 510 contracts + 1 header row
print(df.columns[:6].tolist())       # Filename, then clause column pairs

# inspect one contract
row = df.iloc[1]
print(row['Filename'])
print(row['Governing Law'])              # full clause text
print(row['Governing Law Answer'])       # "Nevada" or "California" etc.
print(row['Non-Compete'])                # clause text or NaN
print(row['Non-Compete Answer'])         # "Yes" or "No" or NaN

# which clause categories are most common across all 510 contracts?
answer_cols = [c for c in df.columns if c.endswith("Answer")]
presence    = (df[answer_cols] == "Yes").sum().sort_values(ascending=False)
print("\nMost common clauses:")
print(presence.head(10))
```

Run the last block to understand which of the 41 categories are richest in the dataset.
The most common categories are the best ones to prioritize in evaluation and demo.

---

### Step C-2 - Build the QA test set

Generate (question, ground_truth, contract_name, clause_category) tuples from
`master_clauses.csv`. These are the inputs to every evaluation step that follows.

`evaluation/build_test_set.py`

```python
import pandas as pd

# These question templates match the CUAD annotation questions exactly
QUESTION_TEMPLATES = {
    "Governing Law":               "Which state or country's law governs this contract?",
    "Non-Compete":                 "Is there a non-compete restriction in this contract?",
    "Termination for Convenience": "Can a party terminate this contract without cause?",
    "Expiration Date":             "On what date does this contract's initial term expire?",
    "Renewal Term":                "What is the renewal term after the initial term expires?",
    "Notice to Terminate Renewal": "What is the notice period required to terminate renewal?",
    "Anti-Assignment":             "Is consent required to assign this contract to a third party?",
    "Change of Control":           "Does a change of control trigger termination or consent rights?",
    "License Grant":               "Does this contract contain a license grant?",
    "Cap on Liability":            "Does this contract include a cap on liability?",
    "Uncapped Liability":          "Is any party's liability uncapped in this contract?",
    "IP Ownership Assignment":     "Does IP created by one party transfer to the counterparty?",
    "Audit Rights":                "Does a party have the right to audit the counterparty?",
    "Insurance":                   "Is there an insurance requirement in this contract?",
    "Liquidated Damages":          "Are there liquidated damages for breach in this contract?",
    # add remaining 26 categories following the same pattern
}

def build_test_set(csv_path="data/master_clauses.csv",
                   output_path="evaluation/eval_test_set.csv"):
    df = pd.read_csv(csv_path)

    test_cases = []
    for _, row in df.iterrows():
        for category, question in QUESTION_TEMPLATES.items():
            answer_col = f"{category} Answer"

            if answer_col not in df.columns:
                continue

            raw_answer = row.get(answer_col, "")

            # NaN means the clause was not found in this contract
            if pd.isna(raw_answer) or str(raw_answer).strip() == "":
                ground_truth = "No"
            else:
                ground_truth = str(raw_answer).strip()

            test_cases.append({
                "contract_name":   row['Filename'],
                "clause_category": category,
                "question":        question,
                "ground_truth":    ground_truth,
            })

    test_df = pd.DataFrame(test_cases)
    test_df.to_csv(output_path, index=False)
    print(f"Test set: {len(test_df)} cases across {test_df['clause_category'].nunique()} categories")
    return test_df

if __name__ == "__main__":
    build_test_set()
```

---

### Step C-3 - Build the EM and F1 scorer

Write and test this independently. These functions are used in every subsequent eval step.

`evaluation/scorer.py`

```python
import re
from collections import Counter

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def exact_match(prediction: str, ground_truth: str) -> int:
    """1 if normalized strings are identical, else 0."""
    return int(normalize(prediction) == normalize(ground_truth))

def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 - same metric used in SQuAD evaluation."""
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common  = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())

    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# quick sanity check
if __name__ == "__main__":
    print(exact_match("Nevada", "Nevada"))          # 1
    print(exact_match("Yes", "No"))                 # 0
    print(token_f1("governed by laws of Nevada",
                   "laws of the State of Nevada"))  # ~0.57
```

---

### Step C-4 - Retrieval evaluation (uses A's Qdrant, no dependency on B)

Before evaluating the LLM output, check whether Person A's index retrieves the correct
chunks. This isolates retrieval failures from generation failures.

`evaluation/retrieval_eval.py`

```python
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient("localhost", port=6333)
model  = SentenceTransformer("BAAI/bge-large-en-v1.5")

def run_retrieval_eval(test_set_path="evaluation/eval_test_set.csv",
                       metadata_path="evaluation/chunks_metadata.csv"):
    test = pd.read_csv(test_set_path)
    meta = pd.read_csv(metadata_path)

    recall_results, mrr_results = [], []

    for _, row in test.iterrows():
        # encode the question
        q_vec = model.encode(row['question'], normalize_embeddings=True).tolist()
        hits  = client.search("cuad_contracts", q_vec, limit=5)
        retrieved_ids = [h.id for h in hits]

        # ground truth: which chunk_ids contain this clause for this contract?
        gt_chunks = meta[
            (meta['contract_name']   == row['contract_name']) &
            (meta['clause_category'] == row['clause_category'])
        ]['chunk_id'].tolist()

        if not gt_chunks:
            continue   # this clause was not present - skip from retrieval eval

        # Recall@5: was any ground truth chunk in the top 5?
        hit = any(r in gt_chunks for r in retrieved_ids)
        recall_results.append(int(hit))

        # MRR: reciprocal rank of the first relevant result
        for rank, rid in enumerate(retrieved_ids):
            if rid in gt_chunks:
                mrr_results.append(1 / (rank + 1))
                break
        else:
            mrr_results.append(0.0)

    print(f"Recall@5 : {sum(recall_results)/len(recall_results):.3f}")
    print(f"MRR      : {sum(mrr_results)/len(mrr_results):.3f}")
    print(f"Evaluated: {len(recall_results)} cases")

if __name__ == "__main__":
    run_retrieval_eval()
```

**Interpret results:** If Recall@5 is below 0.6, the chunking or embedding strategy needs
improvement - bring this finding back to Person A before evaluating generation.

---

### Step C-5 - End-to-end generation evaluation (uses B's /chat API)

Run a sample of the test set through Person B's API and score predictions against ground
truth. Start with 100 cases to check cost and latency before running all 510 contracts.

`evaluation/generation_eval.py`

```python
import requests, time, pandas as pd
from scorer import exact_match, token_f1

def run_generation_eval(test_set_path="evaluation/eval_test_set.csv",
                        output_path="evaluation/eval_results.csv",
                        sample_n=100):
    test   = pd.read_csv(test_set_path)
    sample = test.sample(n=sample_n, random_state=42)

    results = []
    for i, (_, row) in enumerate(sample.iterrows()):
        resp = requests.post("http://localhost:8000/chat", json={
            "question":      row['question'],
            "clause_filter": row['clause_category']
        })
        pred = resp.json().get('answer', '')

        results.append({
            "contract":      row['contract_name'],
            "category":      row['clause_category'],
            "question":      row['question'],
            "ground_truth":  row['ground_truth'],
            "prediction":    pred,
            "em":            exact_match(pred, row['ground_truth']),
            "f1":            token_f1(pred, row['ground_truth']),
        })

        if (i + 1) % 10 == 0:
            print(f"Evaluated {i+1}/{sample_n}")
        time.sleep(0.5)   # avoid rate limiting

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"\nOverall EM : {results_df['em'].mean():.3f}")
    print(f"Overall F1 : {results_df['f1'].mean():.3f}")
    print(f"\nF1 by clause category:")
    print(results_df.groupby('category')[['em','f1']].mean().sort_values('f1'))

    return results_df

if __name__ == "__main__":
    run_generation_eval()
```

---

### Step C-6 - RAGAS faithfulness evaluation

RAGAS checks whether the LLM's answer is actually supported by the retrieved passages.
This catches hallucination even when EM/F1 scores look acceptable.

`evaluation/ragas_eval.py`

```python
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

def run_ragas_eval(results_path="evaluation/eval_results.csv"):
    df = pd.read_csv(results_path)

    ragas_dataset = Dataset.from_dict({
        "question":     df['question'].tolist(),
        "answer":       df['prediction'].tolist(),
        "contexts":     [[gt] for gt in df['ground_truth'].tolist()],
        "ground_truth": df['ground_truth'].tolist(),
    })

    result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )

    print("\nRAGAS scores:")
    print(f"  Faithfulness    : {result['faithfulness']:.3f}")
    print(f"  Answer Relevancy: {result['answer_relevancy']:.3f}")
    print(f"  Context Recall  : {result['context_recall']:.3f}")
    return result

if __name__ == "__main__":
    run_ragas_eval()
```

---

### Step C-7 - Build the evaluation dashboard

Visualize F1 per clause category. This is the primary output for the project report and
demo - it shows exactly which clause types the system handles well and where it fails.

`evaluation/dashboard.py`

```python
import pandas as pd
import matplotlib.pyplot as plt

def build_dashboard(results_path="evaluation/eval_results.csv"):
    df = pd.read_csv(results_path)

    cat_metrics = df.groupby('category')[['em','f1']].mean().sort_values('f1')

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # F1 per clause category
    cat_metrics['f1'].plot(kind='barh', ax=axes[0], color='#5DCAA5')
    axes[0].axvline(cat_metrics['f1'].mean(), color='#D85A30',
                    linestyle='--', label=f"Mean: {cat_metrics['f1'].mean():.2f}")
    axes[0].set_title("Token F1 by clause category")
    axes[0].set_xlabel("F1 score")
    axes[0].legend()

    # EM per clause category
    cat_metrics['em'].plot(kind='barh', ax=axes[1], color='#AFA9EC')
    axes[1].axvline(cat_metrics['em'].mean(), color='#D85A30',
                    linestyle='--', label=f"Mean: {cat_metrics['em'].mean():.2f}")
    axes[1].set_title("Exact match by clause category")
    axes[1].set_xlabel("EM score")
    axes[1].legend()

    plt.suptitle("CUAD RAG Chatbot - Evaluation Results", fontsize=14)
    plt.tight_layout()
    plt.savefig("evaluation/eval_dashboard.png", dpi=150)
    print("Dashboard saved to evaluation/eval_dashboard.png")
    plt.show()

    # error analysis - worst performing cases
    worst = df[df['f1'] < 0.2][['contract','category','question',
                                  'ground_truth','prediction']].head(20)
    worst.to_csv("evaluation/error_analysis.csv", index=False)
    print(f"Error analysis saved - {len(worst)} low-F1 cases")

if __name__ == "__main__":
    build_dashboard()
```

---

### Person C - Final outputs

| File | Description |
|---|---|
| `evaluation/eval_test_set.csv` | All QA test cases with ground truth |
| `evaluation/eval_results.csv` | Predictions + EM + F1 per test case |
| `evaluation/eval_dashboard.png` | Bar charts of F1 and EM by clause category |
| `evaluation/error_analysis.csv` | Low-F1 cases for qualitative analysis |
| RAGAS scores | Faithfulness + relevancy printed to console |

---

---

## Integration Checklist

Run through this checklist when all three modules are ready to connect.

```
[ ] Person A: docker ps shows qdrant/qdrant running on port 6333
[ ] Person A: python ingestion/load_qdrant.py prints vector count > 0
[ ] Person A: bm25_index.pkl exists at ingestion/bm25_index.pkl
[ ] Person A: chunks_metadata.csv exists at evaluation/chunks_metadata.csv

[ ] Person B: curl localhost:8000/health returns {"status":"ok"}
[ ] Person B: curl -X POST localhost:8000/chat -d '{"question":"What is the governing law?"}' returns an answer

[ ] Person C: evaluation/eval_test_set.csv has > 5000 rows
[ ] Person C: python evaluation/scorer.py runs without errors
[ ] Person C: python evaluation/retrieval_eval.py prints Recall@5 and MRR

[ ] Integration: python evaluation/generation_eval.py completes 10 cases
[ ] Integration: evaluation/eval_dashboard.png is generated
```

---

## Installation

```bash
pip install sentence-transformers qdrant-client rank-bm25 openai \
            fastapi uvicorn ragas datasets matplotlib pandas
```

```bash
export OPENAI_API_KEY="your-key-here"
docker run -p 6333:6333 qdrant/qdrant
```

---

## References

- CUAD dataset: https://zenodo.org/records/4595826
- CUAD paper: https://arxiv.org/abs/2103.06268
- BAAI/bge-large-en-v1.5: https://huggingface.co/BAAI/bge-large-en-v1.5
- RAGAS: https://docs.ragas.io
- Qdrant: https://qdrant.tech/documentation
