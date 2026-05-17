FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/     ./data/
COPY src/      ./src/
COPY api/      ./api/
COPY models/   ./models/

RUN python src/train.py --data-path data/WA_Fn-UseC_-Telco-Customer-Churn.csv

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
