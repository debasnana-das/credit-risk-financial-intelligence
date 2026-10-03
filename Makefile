install:
	python -m pip install -r requirements.txt

prepare:
	python -m scripts.prepare_data

credit:
	python -m scripts.train_credit

eps:
	python -m scripts.train_eps

train:
	python -m scripts.train_all

test:
	pytest -q

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn api.main:app --reload
