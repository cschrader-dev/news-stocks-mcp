FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY news_mcp_server.py .
RUN mkdir -p data
COPY data/ data/

EXPOSE 8000

CMD ["python", "news_mcp_server.py"]
