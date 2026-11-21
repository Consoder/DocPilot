"""FastAPI application for a demo-sized, evidence-grounded RAG service."""
from __future__ import annotations
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .core import KnowledgeStore, RAGService, read_document

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=600)
    top_k: int = Field(default=4, ge=1, le=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = os.getenv('DOCPILOT_DB', str(PROJECT_ROOT / 'data' / 'docpilot.sqlite3'))
    store = KnowledgeStore(db_path)
    app.state.store = store
    app.state.rag = RAGService(store)
    yield
    store.close()


app = FastAPI(title='DocPilot - Enterprise RAG Assistant', version='1.0.0', lifespan=lifespan,
              description='Cited hybrid retrieval; optional Gemini generation; retrieval evaluation.')


@app.get('/')
def home():
    return FileResponse(PROJECT_ROOT / 'web' / 'index.html')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/api/status')
def status():
    return {'generation_enabled': app.state.rag.generator.enabled,
            'embedding_backend': app.state.store.embedder.name,
            'documents': len(app.state.store.list_documents())}


@app.get('/api/documents')
def documents():
    return app.state.store.list_documents()


@app.post('/api/documents', status_code=201)
async def upload(file: UploadFile = File(...)):
    filename = Path(file.filename or '').name
    if not filename or Path(filename).suffix.lower() not in {'.pdf', '.txt', '.md'}:
        raise HTTPException(status_code=400, detail='Upload a PDF, TXT, or MD file')
    data = await file.read(5 * 1024 * 1024 + 1)
    try:
        text = read_document(filename, data)
        return app.state.store.add_document(filename, file.content_type or '', text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete('/api/documents/{doc_id}')
def delete_document(doc_id: str):
    if not app.state.store.delete_document(doc_id):
        raise HTTPException(status_code=404, detail='Document not found')
    return {'deleted': True}


@app.post('/api/ask')
def ask(body: Question):
    try:
        return app.state.rag.ask(body.question, body.top_k)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post('/api/demo/seed')
def seed_demo():
    existing = {d['title'] for d in app.state.store.list_documents()}
    result = []
    for path in sorted((PROJECT_ROOT / 'sample_docs').glob('*.md')):
        if path.name not in existing:
            result.append(app.state.store.add_document(path.name, 'text/markdown', path.read_text(encoding='utf-8')))
    return {'added': result, 'message': 'Sample company and policies are fictional.'}


@app.get('/api/evaluation')
def evaluation():
    fixture_path = PROJECT_ROOT / 'evals' / 'questions.json'
    data = json.loads(fixture_path.read_text(encoding='utf-8'))
    matched1, matched3, matched5, reciprocal, details = 0, 0, 0, 0.0, []
    for item in data:
        hits = app.state.store.search(item['question'], k=5)
        order = [h.document_title for h in hits]
        found = next((i + 1 for i, title in enumerate(order) if title == item['expected_document']), None)
        matched1 += int(found == 1)
        matched3 += int(found is not None and found <= 3)
        matched5 += int(found is not None)
        reciprocal += 1 / found if found else 0
        details.append({'question': item['question'], 'expected': item['expected_document'],
                        'retrieved': order, 'hit': bool(found), 'rank': found})
    n = len(data)
    return {'total': n, 'hit_at_1': round(matched1 / n, 3),
            'hit_at_3': round(matched3 / n, 3), 'hit_at_5': round(matched5 / n, 3), 'mrr_at_5': round(reciprocal / n, 3),
            'details': details, 'note': 'Retrieval quality only; not an LLM-answer accuracy benchmark.'}
