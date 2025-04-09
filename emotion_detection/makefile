install:
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

run-dev:
	fastapi dev main.py --port 8000