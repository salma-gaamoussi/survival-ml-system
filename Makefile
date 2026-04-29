DATA_PATH = data/WA_Fn-UseC_-Telco-Customer-Churn.csv
IMAGE     = survival-api

.PHONY: install train test serve build run clean

install:
	pip install -r requirements.txt

train:
	python src/train.py --data-path $(DATA_PATH)

test:
	pytest tests/ -v

serve:           ## run API locally (no Docker)
	uvicorn api.main:app --reload --port 8000

build:           ## build Docker image
	docker build -t $(IMAGE) .

run:             ## run containerised API
	docker run -p 8000:8000 $(IMAGE)

clean:
	rm -f models/*.pkl models/*.pt