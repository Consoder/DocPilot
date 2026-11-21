FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY docpilot ./docpilot
COPY web ./web
COPY sample_docs ./sample_docs
COPY evals ./evals
RUN mkdir -p /app/data && useradd -u 10001 -m appuser && chown appuser:appuser /app/data
USER appuser
ENV DOCPILOT_DB=/app/data/docpilot.sqlite3 PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "docpilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
