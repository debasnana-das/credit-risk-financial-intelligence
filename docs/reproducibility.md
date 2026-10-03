# Reproducibility

1. Put the supplied source workbooks in `data/raw/`.
2. Create an environment with Python 3.11+.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Prepare CSVs:

```bash
python scripts/prepare_data.py
```

5. Train:

```bash
python scripts/train_credit.py
python scripts/train_eps.py
```

6. Run tests:

```bash
pytest -q
```

7. Launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

8. Launch the API:

```bash
uvicorn api.main:app --reload
```
