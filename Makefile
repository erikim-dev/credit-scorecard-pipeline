.PHONY: setup data eda train evaluate serve monitor test clean all

# ─── Configuration ────────────────────────────────────────────
PYTHON      := python
PIP         := pip
DATA_DIR    := data/raw
PROCESSED   := data/processed
NOTEBOOKS   := notebooks
API_HOST    := 0.0.0.0
API_PORT    := 8000

# ─── Setup ────────────────────────────────────────────────────
setup:
	$(PIP) install -r requirements.txt

# ─── Data ─────────────────────────────────────────────────────
data:
	kaggle competitions download -c home-credit-default-risk
	unzip -o home-credit-default-risk.zip -d $(DATA_DIR)/
	@echo "Data downloaded to $(DATA_DIR)/"

# ─── Feature Engineering (SQL + Python) ──────────────────────
features:
	$(PYTHON) src/feature_engineering.py --input $(DATA_DIR) --output $(PROCESSED)
	@echo "Features written to $(PROCESSED)/"

# ─── Train Models ─────────────────────────────────────────────
train-champion:
	$(PYTHON) src/train.py --model scorecard --data $(PROCESSED)
	@echo "Champion scorecard trained"

train-challenger:
	$(PYTHON) src/train.py --model xgboost --data $(PROCESSED)
	@echo "Challenger XGBoost trained"

train-lightgbm:
	$(PYTHON) src/train.py --model lightgbm --data $(PROCESSED)
	@echo "Challenger LightGBM trained"

train-stacking:
	$(PYTHON) src/train.py --model stacking --data $(PROCESSED)
	@echo "Stacking ensemble trained"

train: train-champion train-challenger train-lightgbm train-stacking

# ─── Evaluate ─────────────────────────────────────────────────
evaluate:
	$(PYTHON) src/evaluate.py --models models/ --data $(PROCESSED)
	@echo "Evaluation complete -- see reports/"

# ─── Serve API ────────────────────────────────────────────────
serve:
	uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) --reload

# ─── Monitor ──────────────────────────────────────────────────
monitor:
	$(PYTHON) monitoring/psi_monitor.py --baseline $(PROCESSED)/train_features.csv --current $(PROCESSED)/latest.csv

# ─── Test ─────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ─── Docker ───────────────────────────────────────────────────
docker-build:
	docker build -t credit-risk-api .

docker-run:
	docker run -p $(API_PORT):$(API_PORT) credit-risk-api

# ─── MLflow UI ────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --backend-store-uri mlruns/

# ─── Clean ────────────────────────────────────────────────────
clean:
	rm -rf $(PROCESSED)/*.csv models/*.pkl mlruns/* __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	@echo "Cleaned"

# ─── Full Pipeline ─────────────────────────────────────────────
all: setup data features train evaluate
	@echo "Full pipeline complete"
