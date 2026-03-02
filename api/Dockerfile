FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install -r requirements.txt

COPY api/src/    ./src/
COPY api/models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
