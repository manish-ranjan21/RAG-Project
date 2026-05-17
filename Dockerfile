FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer cache — only reruns if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and config
COPY src/ ./src/
COPY .env.example .env

# Persistent directories (mount these as volumes in production)
RUN mkdir -p src/docs src/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "src/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
