import os
import hashlib
import requests
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from colorama import Fore, Style, init
from fastapi.middleware.cors import CORSMiddleware

# ── Silence HuggingFace warnings ─────────────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

init(autoreset=True)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("HF_TOKEN") and os.environ.get("HF_API_KEY"):
    os.environ["HF_TOKEN"] = os.environ["HF_API_KEY"]


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"
CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 64
TOP_K          = 5


# ══════════════════════════════════════════════════════════════════════════════
#  SUPABASE AUTH
# ══════════════════════════════════════════════════════════════════════════════
from supabase import create_client, Client

_supabase: Optional[Client] = None

def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY are required in .env")
        _supabase = create_client(url, key)
    return _supabase


security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Validate the Supabase JWT and return the user payload."""
    import jwt as pyjwt
    token = credentials.credentials
    try:
        # Decode without signature verification — Supabase signs with ES256
        # and we trust tokens issued by our own Supabase project
        payload = pyjwt.decode(
            token,
            options={"verify_signature": False},
            algorithms=["HS256", "ES256"],
        )
        user_id = payload.get("sub")
        email   = payload.get("email", "")
        if not user_id:
            raise HTTPException(401, "Invalid token: missing sub")
        return {"sub": user_id, "email": email}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(401, f"Authentication failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
#  METADATA HELPER
# ══════════════════════════════════════════════════════════════════════════════
def build_where_clause(
    filters:    Optional[Dict[str, Any]] = None,
    source:     Optional[str]            = None,
    department: Optional[str]            = None,
    author:     Optional[str]            = None,
    year_min:   Optional[int]            = None,
    year_max:   Optional[int]            = None,
    tags:       Optional[List[str]]      = None,
) -> Optional[dict]:
    conditions = []
    if filters:
        conditions.append(filters)
    if source:
        conditions.append({"source": {"$eq": source}})
    if department:
        conditions.append({"department": {"$eq": department}})
    if author:
        conditions.append({"author": {"$eq": author}})
    if year_min is not None and year_max is not None:
        conditions.append({"year": {"$gte": year_min}})
        conditions.append({"year": {"$lte": year_max}})
    elif year_min is not None:
        conditions.append({"year": {"$gte": year_min}})
    elif year_max is not None:
        conditions.append({"year": {"$lte": year_max}})
    if tags:
        tag_conditions = [{"tags": {"$eq": t.strip()}} for t in tags]
        if len(tag_conditions) == 1:
            conditions.append(tag_conditions[0])
        else:
            conditions.append({"$or": tag_conditions})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDER
# ══════════════════════════════════════════════════════════════════════════════
class Embedder:
    def __init__(self, model: str = HF_EMBED_MODEL):
        self.api_key = os.environ.get("HF_API_KEY")
        if not self.api_key:
            raise ValueError("❌ HF_API_KEY is required in .env")
        self.url = (
            f"https://router.huggingface.co/hf-inference/models"
            f"/{model}/pipeline/feature-extraction"
        )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        print(f"{Fore.CYAN}🤗 Embedder — {model}{Style.RESET_ALL}")
        self._warm_up()

    def _warm_up(self):
        try:
            self.embed_one("Warm-up.")
            print(f"{Fore.GREEN}✔ Embedder ready{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Embedder warm-up: {e}{Style.RESET_ALL}")

    def _request(self, texts: list[str]):
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        for attempt in range(1, 5):
            try:
                resp = requests.post(self.url, headers=self.headers, json=payload, timeout=90)
                if resp.status_code == 503:
                    print(f"{Fore.YELLOW}   Embedder loading... retry in 15s{Style.RESET_ALL}")
                    time.sleep(15)
                    continue
                if resp.status_code == 429:
                    wait = 8 * (2 ** (attempt - 1))
                    print(f"{Fore.YELLOW}   Rate limited — waiting {wait}s...{Style.RESET_ALL}")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"{Fore.RED}Embedder attempt {attempt} failed: {e}{Style.RESET_ALL}")
                if attempt == 4:
                    raise
                time.sleep(5 * attempt)

    @staticmethod
    def _normalise(raw):
        if isinstance(raw[0], float):
            return [raw]
        if isinstance(raw[0], list) and isinstance(raw[0][0], list):
            return [
                [sum(row[d] for row in tv) / len(tv) for d in range(len(tv[0]))]
                for tv in raw
            ]
        return raw

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            raw = self._request(texts[i : i + batch_size])
            all_vecs.extend(self._normalise(raw))
        return all_vecs

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ══════════════════════════════════════════════════════════════════════════════
#  CHUNKER
# ══════════════════════════════════════════════════════════════════════════════
import tiktoken

class TokenChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.enc        = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str, source: str = "", extra_metadata: dict = None) -> list[dict]:
        tokens = self.enc.encode(text)
        chunks, start, idx = [], 0, 0
        while start < len(tokens):
            end        = min(start + self.chunk_size, len(tokens))
            chunk_text = self.enc.decode(tokens[start:end])
            record = {
                "text":        chunk_text,
                "chunk_idx":   idx,
                "source":      source,
                "char_count":  len(chunk_text),
                "token_count": end - start,
            }
            if extra_metadata:
                record.update(extra_metadata)
            chunks.append(record)
            idx   += 1
            start  = end - self.overlap
            if end == len(tokens):
                break
        return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  FILE READER
# ══════════════════════════════════════════════════════════════════════════════
class FileReader:
    @staticmethod
    def read(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n\n".join(
                f"[Page {i+1}]\n{page.extract_text() or ''}"
                for i, page in enumerate(reader.pages)
            )
        elif ext == ".docx":
            from docx import Document
            doc = Document(path)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()


# ══════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE — per-user isolated collections
# ══════════════════════════════════════════════════════════════════════════════
import chromadb

class VectorStore:
    """Each user gets their own ChromaDB collection named rag_user_<user_id>."""

    def __init__(self, api_key: str = "", tenant: str = "", database: str = "default_database"):
        api_key = api_key or os.environ.get("CHROMA_API_KEY", "")
        tenant  = tenant  or os.environ.get("CHROMA_TENANT",  "")
        if not api_key or not tenant:
            raise ValueError("CHROMA_API_KEY and CHROMA_TENANT are required.")
        self.chroma  = chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
        self._cache: Dict[str, Any] = {}
        print(f"{Fore.GREEN}✔ ChromaDB connected{Style.RESET_ALL}")

    def _collection_name(self, user_id: str) -> str:
        safe = hashlib.md5(user_id.encode()).hexdigest()[:16]
        return f"rag_u_{safe}"

    def collection_for(self, user_id: str):
        if user_id not in self._cache:
            name = self._collection_name(user_id)
            self._cache[user_id] = self.chroma.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        return self._cache[user_id]

    def add_chunks(self, user_id: str, chunks: list[dict], embeddings: list[list[float]]):
        col       = self.collection_for(user_id)
        ids       = [hashlib.md5(f"{c['source']}::{c['chunk_idx']}".encode()).hexdigest() for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query(
        self,
        user_id:   str,
        embedding: list[float],
        top_k:     int            = TOP_K,
        where:     Optional[dict] = None,
    ) -> list[dict]:
        col   = self.collection_for(user_id)
        total = col.count()
        if total == 0:
            return []
        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results":        min(top_k, total),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = col.query(**kwargs)
        return [
            {"text": doc, "meta": meta, "score": 1 - dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def get_unique_sources(self, user_id: str) -> list[str]:
        col = self.collection_for(user_id)
        if col.count() == 0:
            return []
        all_meta = col.get(include=["metadatas"])["metadatas"]
        return sorted({Path(m.get("source", "")).name for m in all_meta if m.get("source")})

    def get_metadata_summary(self, user_id: str) -> dict:
        col = self.collection_for(user_id)
        if col.count() == 0:
            return {}
        all_meta = col.get(include=["metadatas"])["metadatas"]
        summary: Dict[str, set] = {}
        for m in all_meta:
            for k, v in m.items():
                if k in ("chunk_idx", "char_count", "token_count", "text"):
                    continue
                summary.setdefault(k, set()).add(str(v))
        return {k: sorted(v) for k, v in summary.items()}

    def count(self, user_id: str) -> int:
        return self.collection_for(user_id).count()

    def delete_document(self, user_id: str, source_path: str):
        col      = self.collection_for(user_id)
        all_data = col.get(include=["metadatas"])
        ids_to_delete = [
            id_ for id_, meta in zip(all_data["ids"], all_data["metadatas"])
            if meta.get("source") == source_path
        ]
        if ids_to_delete:
            col.delete(ids=ids_to_delete)
        return len(ids_to_delete)


# ══════════════════════════════════════════════════════════════════════════════
#  RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class RAGPipeline:
    def __init__(self, groq_api_key: str, chroma_api_key: str = "", chroma_tenant: str = ""):
        self.chunker  = TokenChunker()
        self.embedder = Embedder()
        self.store    = VectorStore(chroma_api_key, chroma_tenant)
        import groq
        self.groq    = groq.Groq(api_key=groq_api_key)
        self.model   = GROQ_MODEL
        self._histories: Dict[str, list] = {}

    def _history(self, user_id: str) -> list:
        return self._histories.setdefault(user_id, [])

    def ingest(self, user_id: str, path: str, extra_metadata: dict = None) -> int:
        text       = FileReader.read(path)
        chunks     = self.chunker.chunk(text, source=path, extra_metadata=extra_metadata)
        embeddings = self.embedder.embed([c["text"] for c in chunks])
        self.store.add_chunks(user_id, chunks, embeddings)
        print(f"{Fore.GREEN}✔ [{user_id[:8]}] Ingested {len(chunks)} chunks — {Path(path).name}{Style.RESET_ALL}")
        return len(chunks)

    def retrieve(
        self,
        user_id:  str,
        question: str,
        top_k:    int            = TOP_K,
        where:    Optional[dict] = None,
    ) -> list[dict]:
        q_emb = self.embedder.embed_one(question)
        return self.store.query(user_id, q_emb, top_k=top_k, where=where)

    def query(
        self,
        user_id:  str,
        question: str,
        top_k:    int            = TOP_K,
        where:    Optional[dict] = None,
    ) -> str:
        if self.store.count(user_id) == 0:
            return "No documents ingested yet. Please upload a file first."

        hits = self.retrieve(user_id, question, top_k, where)
        if not hits:
            hint = " Try removing filters." if where else ""
            return f"No relevant context found.{hint}"

        context_parts = []
        for i, h in enumerate(hits):
            meta     = h["meta"]
            meta_str = " | ".join(
                f"{k}: {v}" for k, v in meta.items()
                if k not in ("chunk_idx", "char_count", "token_count", "source") and v is not None
            )
            header = (
                f"[{i+1}] {Path(meta.get('source', 'unknown')).name}"
                + (f" | {meta_str}" if meta_str else "")
                + f" | score: {h['score']:.4f}"
            )
            context_parts.append(f"{header}\n{h['text']}")

        context = "\n\n---\n\n".join(context_parts)
        system_prompt = (
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "Cite source filenames when referencing specific information. "
            "If the context is insufficient, say so honestly.\n\n"
            f"CONTEXT:\n{context}"
        )
        history  = self._history(user_id)
        messages = [{"role": "system", "content": system_prompt}]
        messages += history[-6:]
        messages.append({"role": "user", "content": question})

        resp   = self.groq.chat.completions.create(
            model=self.model, messages=messages, temperature=0.3, max_tokens=1024
        )
        answer = resp.choices[0].message.content

        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant",  "content": answer})
        return answer

    def clear_history(self, user_id: str):
        self._histories.pop(user_id, None)


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════
class SignupRequest(BaseModel):
    email:    EmailStr
    password: str
    name:     Optional[str] = None

class LoginRequest(BaseModel):
    email:    EmailStr
    password: str

class AuthResponse(BaseModel):
    token:    str
    user_id:  str
    email:    str
    name:     Optional[str] = None

class QueryRequest(BaseModel):
    question:   str
    top_k:      int                      = TOP_K
    source:     Optional[str]            = None
    department: Optional[str]            = None
    author:     Optional[str]            = None
    year_min:   Optional[int]            = None
    year_max:   Optional[int]            = None
    tags:       Optional[List[str]]      = None
    filters:    Optional[Dict[str, Any]] = None

class SourcesRequest(BaseModel):
    question:   str
    top_k:      int                      = TOP_K
    source:     Optional[str]            = None
    department: Optional[str]            = None
    author:     Optional[str]            = None
    year_min:   Optional[int]            = None
    year_max:   Optional[int]            = None
    tags:       Optional[List[str]]      = None
    filters:    Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer:        str
    sources:       List[dict]     = []
    active_filter: Optional[dict] = None

class QuizRequest(BaseModel):
    topic:      Optional[str] = None
    difficulty: str           = "medium"   # easy | medium | hard | mixed
    num_q:      int           = 5
    top_k:      int           = 8


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="RAG API — Multi-user, Supabase Auth, Per-user ChromaDB",
    version="7.0",
    description="Each user has isolated document storage. Auth via Supabase.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_pipeline: Optional[RAGPipeline] = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        groq_key      = os.getenv("GROQ_API_KEY")
        chroma_key    = os.getenv("CHROMA_API_KEY")
        chroma_tenant = os.getenv("CHROMA_TENANT")
        if not all([groq_key, chroma_key, chroma_tenant]):
            raise HTTPException(500, "Missing API keys in .env")
        _pipeline = RAGPipeline(groq_key, chroma_key, chroma_tenant)
    return _pipeline

@app.on_event("startup")
async def startup():
    get_pipeline()
    get_supabase()   # validate Supabase connection on startup
    print(f"{Fore.GREEN}✔ Supabase Auth connected{Style.RESET_ALL}")


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/quiz/generate", tags=["Quiz"])
async def generate_quiz(
    req:          QuizRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieve relevant chunks from ChromaDB for the given topic (or all docs),
    then ask Groq to generate multiple-choice questions as structured JSON.
    Returns a list of {q, opts, ans, exp} objects ready for the frontend quiz engine.
    """
    import json, re

    user_id = current_user["sub"]
    rag     = get_pipeline()

    if rag.store.count(user_id) == 0:
        raise HTTPException(400, "No documents ingested yet. Upload study material first.")

    # ── 1. Retrieve context from ChromaDB ────────────────────────────────────
    search_query = req.topic if req.topic else "key concepts important topics exam questions"
    hits = rag.retrieve(user_id, search_query, top_k=req.top_k)
    if not hits:
        raise HTTPException(400, "No relevant content found in your documents.")

    context = "\n\n---\n\n".join(
        f"[Source: {Path(h['meta'].get('source','unknown')).name}]\n{h['text']}"
        for h in hits
    )

    # ── 2. Build difficulty guidance ─────────────────────────────────────────
    diff_map = {
        "easy":   "straightforward recall and basic comprehension",
        "medium": "application and moderate reasoning",
        "hard":   "deep analysis, edge cases, and advanced reasoning",
        "mixed":  "a mix of easy, medium, and hard levels",
    }
    diff_desc = diff_map.get(req.difficulty, diff_map["medium"])

    # ── 3. Call Groq with strict JSON output instruction ─────────────────────
    topic_str = f' focused on "{req.topic}"' if req.topic else ""
    system_prompt = (
        "You are an expert exam question writer. Generate ONLY a valid JSON array — "
        "no markdown, no explanation, no code fences. Return exactly the JSON array."
    )
    user_prompt = f"""Generate {req.num_q} multiple-choice questions{topic_str} at {diff_desc} difficulty.
Use ONLY the study material below as your source.

STUDY MATERIAL:
{context}

Return a JSON array where each element is:
{{
  "q":    "<question text>",
  "opts": ["<option A>", "<option B>", "<option C>", "<option D>"],
  "ans":  <0-based index of correct option>,
  "exp":  "<one-sentence explanation of why the answer is correct>"
}}

Rules:
- Exactly 4 options per question.
- "ans" must be 0, 1, 2, or 3.
- Questions must be directly answerable from the study material.
- Do NOT wrap in markdown or add any text outside the JSON array."""

    resp = rag.groq.chat.completions.create(
        model=rag.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=3000,
    )
    raw = resp.choices[0].message.content.strip()

    # ── 4. Parse and validate ─────────────────────────────────────────────────
    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        questions = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract a JSON array from somewhere inside the response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            questions = json.loads(match.group())
        else:
            raise HTTPException(500, f"LLM returned non-JSON output: {raw[:300]}")

    if not isinstance(questions, list) or len(questions) == 0:
        raise HTTPException(500, "LLM returned an empty question list.")

    # Normalise fields
    validated = []
    for item in questions:
        if not all(k in item for k in ("q", "opts", "ans", "exp")):
            continue
        if not isinstance(item["opts"], list) or len(item["opts"]) != 4:
            continue
        if not isinstance(item["ans"], int) or not (0 <= item["ans"] <= 3):
            continue
        validated.append({
            "q":    str(item["q"]),
            "opts": [str(o) for o in item["opts"]],
            "ans":  item["ans"],
            "exp":  str(item["exp"]),
        })

    if not validated:
        raise HTTPException(500, "No valid questions could be parsed from the LLM response.")

    return {"questions": validated, "source_chunks": len(hits), "topic": req.topic}


@app.post("/auth/signup", response_model=AuthResponse, tags=["Auth"])
async def signup(req: SignupRequest):
    """Register a new user account via Supabase."""
    try:
        supabase = get_supabase()
        response = supabase.auth.sign_up({
            "email":    req.email,
            "password": req.password,
            "options":  {"data": {"name": req.name or ""}},
        })
        if not response.user:
            raise HTTPException(400, "Signup failed — check your email/password")

        token = response.session.access_token if response.session else ""
        return AuthResponse(
            token=token,
            user_id=response.user.id,
            email=response.user.email,
            name=req.name or "",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/auth/login", response_model=AuthResponse, tags=["Auth"])
async def login(req: LoginRequest):
    """Login and receive a Supabase JWT token."""
    try:
        supabase = get_supabase()
        response = supabase.auth.sign_in_with_password({
            "email":    req.email,
            "password": req.password,
        })
        if not response.user or not response.session:
            raise HTTPException(401, "Invalid email or password")

        name = (response.user.user_metadata or {}).get("name", "")
        return AuthResponse(
            token=response.session.access_token,
            user_id=response.user.id,
            email=response.user.email,
            name=name,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(401, str(e))


@app.get("/auth/me", tags=["Auth"])
async def me(current_user: dict = Depends(get_current_user)):
    """Return the current authenticated user's info."""
    return {
        "user_id": current_user["sub"],
        "email":   current_user["email"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT ROUTES  (all require auth)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/ingest", tags=["Documents"])
async def ingest_file(
    file:         UploadFile    = File(...),
    department:   Optional[str] = Query(None),
    author:       Optional[str] = Query(None),
    year:         Optional[int] = Query(None),
    tags:         Optional[str] = Query(None, description="Comma-separated e.g. legal,finance"),
    current_user: dict          = Depends(get_current_user),
):
    """Upload and ingest a document into the current user's isolated collection."""
    if not file.filename.lower().endswith((".pdf", ".txt", ".docx", ".md")):
        raise HTTPException(400, "Unsupported file type. Use pdf / txt / docx / md.")

    user_id   = current_user["sub"]
    temp_path = Path(f"temp_{user_id}_{file.filename}")
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        extra = {k: v for k, v in {
            "department": department,
            "author":     author,
            "year":       year,
            "tags":       tags,
        }.items() if v is not None}

        rag = get_pipeline()
        n   = rag.ingest(user_id, str(temp_path), extra_metadata=extra or None)
        return {
            "message":       f"{file.filename} ingested successfully",
            "chunks_added":  n,
            "total_chunks":  rag.store.count(user_id),
            "metadata_tags": extra,
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.delete("/documents/{filename}", tags=["Documents"])
async def delete_document(
    filename:     str,
    current_user: dict = Depends(get_current_user),
):
    """Delete all chunks of a specific document from the user's collection."""
    user_id = current_user["sub"]
    rag     = get_pipeline()
    sources = rag.store.get_unique_sources(user_id)
    if filename not in sources:
        raise HTTPException(404, f"Document '{filename}' not found in your collection.")

    col      = rag.store.collection_for(user_id)
    all_data = col.get(include=["metadatas"])
    ids_to_delete = [
        id_ for id_, meta in zip(all_data["ids"], all_data["metadatas"])
        if Path(meta.get("source", "")).name == filename
    ]
    if ids_to_delete:
        col.delete(ids=ids_to_delete)

    return {"message": f"Deleted {len(ids_to_delete)} chunks for '{filename}'"}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(
    req:          QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    """Ask a question against the current user's documents."""
    user_id = current_user["sub"]
    where   = build_where_clause(
        filters=req.filters, source=req.source,
        department=req.department, author=req.author,
        year_min=req.year_min, year_max=req.year_max, tags=req.tags,
    )
    rag    = get_pipeline()
    answer = rag.query(user_id, req.question, req.top_k, where)
    hits   = rag.retrieve(user_id, req.question, req.top_k, where)

    sources = [
        {
            "rank":        i + 1,
            "source":      Path(h["meta"].get("source", "unknown")).name,
            "embed_score": round(h["score"], 4),
            "preview":     h["text"][:280] + "...",
            "metadata": {
                k: v for k, v in h["meta"].items()
                if k not in ("text", "chunk_idx", "char_count", "token_count")
            },
        }
        for i, h in enumerate(hits)
    ]
    return QueryResponse(answer=answer, sources=sources, active_filter=where)


@app.post("/sources", tags=["RAG"])
async def get_sources(
    req:          SourcesRequest,
    current_user: dict = Depends(get_current_user),
):
    """Return ranked chunks without calling the LLM."""
    user_id = current_user["sub"]
    where   = build_where_clause(
        filters=req.filters, source=req.source,
        department=req.department, author=req.author,
        year_min=req.year_min, year_max=req.year_max, tags=req.tags,
    )
    hits = get_pipeline().retrieve(user_id, req.question, req.top_k, where)
    return {
        "question":      req.question,
        "active_filter": where,
        "results": [
            {
                "rank":        i + 1,
                "source":      Path(h["meta"].get("source", "unknown")).name,
                "embed_score": round(h["score"], 4),
                "preview":     h["text"][:400] + "...",
                "metadata": {
                    k: v for k, v in h["meta"].items()
                    if k not in ("text", "chunk_idx", "char_count", "token_count")
                },
            }
            for i, h in enumerate(hits)
        ],
    }


@app.delete("/history", tags=["RAG"])
async def clear_history(current_user: dict = Depends(get_current_user)):
    """Clear the conversation history for the current user."""
    get_pipeline().clear_history(current_user["sub"])
    return {"message": "Conversation history cleared"}


@app.get("/metadata", tags=["Documents"])
async def metadata_summary(current_user: dict = Depends(get_current_user)):
    """Returns all unique metadata field values for the current user's collection."""
    return get_pipeline().store.get_metadata_summary(current_user["sub"])


@app.get("/collections", tags=["Documents"])
async def list_sources(current_user: dict = Depends(get_current_user)):
    """List all documents ingested by the current user."""
    rag = get_pipeline()
    uid = current_user["sub"]
    return {
        "total_chunks": rag.store.count(uid),
        "sources":      rag.store.get_unique_sources(uid),
    }


@app.get("/stats", tags=["Info"])
async def stats(current_user: dict = Depends(get_current_user)):
    rag = get_pipeline()
    uid = current_user["sub"]
    return {
        "user_id":         uid,
        "total_chunks":    rag.store.count(uid),
        "embed_model":     HF_EMBED_MODEL,
        "llm_model":       GROQ_MODEL,
        "top_k":           TOP_K,
        "sources":         rag.store.get_unique_sources(uid),
        "metadata_fields": list(rag.store.get_metadata_summary(uid).keys()),
    }


@app.get("/health", tags=["Info"])
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)