from pathlib import Path
from fastapi.testclient import TestClient
from docpilot.api import app


def test_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setenv('DOCPILOT_DB', str(tmp_path / 'api.sqlite3'))
    monkeypatch.setenv('EMBEDDING_BACKEND', 'hashing')
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    with TestClient(app) as client:
        assert client.get('/health').json() == {'status': 'ok'}
        assert client.get('/api/documents').json() == []
        seed = client.post('/api/demo/seed')
        assert seed.status_code == 200
        assert len(seed.json()['added']) == 4
        answer = client.post('/api/ask', json={'question': 'How are API keys rotated?'})
        assert answer.status_code == 200
        assert answer.json()['sources']
        assert answer.json()['mode'] == 'retrieval_only'
        assert client.post('/api/ask', json={'question': 'X'}).status_code == 422
        evaluation = client.get('/api/evaluation').json()
        assert evaluation['total'] == 8
        assert evaluation['hit_at_1'] >= 0.75
        assert evaluation['mrr_at_5'] >= 0.85
        assert client.post('/api/documents', files={'file': ('evil.exe', b'hello')}).status_code == 400
        addition = client.post('/api/documents', files={'file': ('custom.md', b'Custom fact: 123XYZ')})
        assert addition.status_code == 201
        assert client.delete(f"/api/documents/{addition.json()['id']}").status_code == 200
        assert len(client.get('/api/documents').json()) == 4
