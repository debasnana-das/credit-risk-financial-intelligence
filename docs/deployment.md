# Deployment

## Streamlit Community Cloud — primary

1. Push the repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Create an app from your repository.
4. Use `app/streamlit_app.py` as the entrypoint.
5. Commit/push changes and let the deployment update from GitHub.

Official documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Render — optional API

Use the included `render.yaml`, or create a Python Web Service with:

```text
Build: pip install -r requirements.txt
Start: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Official documentation: https://render.com/docs/deploy-fastapi
