# backend/routers/query.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import requests
import chromadb
from dotenv import load_dotenv
from typing import Any

router = APIRouter()


# Load environment variables
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_CLOUD_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_CLOUD_TENANT")
CHROMA_DB = "HeatX Software"
CHROMA_COLLECTION = "Heatx"

if not PERPLEXITY_API_KEY or not CHROMA_API_KEY or not CHROMA_TENANT:
    raise RuntimeError("❌ Missing required API keys/tenant in environment variables.")

client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DB,
)

try:
    knowledge_collection = client.get_collection(CHROMA_COLLECTION)
except Exception as e:
    raise RuntimeError(f"❌ Failed to access ChromaDB collection '{CHROMA_COLLECTION}': {e}")

class QueryRequest(BaseModel):
    query: str
    n_results: int = 3

class QueryResponse(BaseModel):
    answer: str

def query_chroma_collection(query: str, n_results: int = 3) -> str:
    results = knowledge_collection.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""
    return "\n\n".join(docs)

def call_perplexity_api(query: str, context: str) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": f"Context:\n{context}\n\nUser Question:\n{query}"}
        ],
        "temperature": 0.4,
        "max_tokens": 1024,
        "stream": False,
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Perplexity API error: {response.status_code} {response.text}")

@router.post("/", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    try:
        context = query_chroma_collection(request.query, request.n_results)
        if not context.strip():
            return QueryResponse(answer="❌ No relevant information found in the knowledge base.")
        response_text = call_perplexity_api(request.query, context)
        return QueryResponse(answer=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
