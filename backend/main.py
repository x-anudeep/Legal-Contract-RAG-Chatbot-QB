from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .generator import generate
from .reranker import rerank
from .retriever import retrieve


app = FastAPI(title="CUAD RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    clause_filter: str | None = None


class RetrieveRequest(ChatRequest):
    top_k: int = Field(default=5, ge=1, le=50)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


class RetrieveResponse(BaseModel):
    results: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest) -> dict[str, list[dict[str, Any]]]:
    candidates = retrieve(req.question, req.clause_filter, top_k=req.top_k)
    return {"results": candidates}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict[str, Any]:
    candidates = retrieve(req.question, req.clause_filter, top_k=20)
    top_chunks = rerank(req.question, candidates, top_k=5)
    return generate(req.question, top_chunks)
