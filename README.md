# PDF Chat AI Agent

Upload a PDF and ask questions about it. Built with Streamlit and the Gemini API.

## Run locally

```bash
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and put your NEW Gemini API key
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository and push these files to it
   (`.streamlit/secrets.toml` is git-ignored, so your key stays private).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** -> choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings -> Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "your-new-api-key-here"
   ```

5. Click **Deploy**. You get a public URL like `https://your-app.streamlit.app`.

## Notes

- Works only with PDFs that contain selectable text (not scanned images).
- Change the model by adding `GEMINI_MODEL = "..."` to your secrets.
