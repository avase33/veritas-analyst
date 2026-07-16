.PHONY: install dev test lint demo serve frontend docker clean

install:
	pip install -e .

dev:
	pip install -e ".[serve,dev]"

test:
	pytest --cov=veritas --cov-report=term-missing

lint:
	ruff check veritas scripts

demo:
	python -m veritas demo

serve:
	veritas serve

frontend:
	cd frontend && npm install && npm run dev

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache build dist *.egg-info sample_docs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
