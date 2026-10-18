import io
import os

import streamlit as st
from google import genai
from pypdf import PdfReader

# --- 1. Page config ---
st.set_page_config(page_title="PDF Chat AI Agent", page_icon="🤖", layout="wide")


# --- 2. Credentials (never hardcode keys) ---
def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def get_model_name():
    try:
        if "GEMINI_MODEL" in st.secrets:
            return st.secrets["GEMINI_MODEL"]
    except Exception:
        pass
    return os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


def ask_gemini(client, model, prompt):
    """Use the Interactions API (recommended); fall back to generate_content."""
    try:
        interaction = client.interactions.create(model=model, input=prompt)
        return interaction.output_text
    except AttributeError:
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text


API_KEY = get_api_key()
MODEL = get_model_name()

# --- 3. Sidebar ---
st.sidebar.title("📁 Document Panel")
st.sidebar.write("Upload your document here to start chatting with it.")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# Reset chat when a new file is uploaded
if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_file:
    st.session_state.messages = []
    st.session_state.last_uploaded_file = uploaded_file.name

if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()


# --- 4. PDF text extraction ---
@st.cache_data(show_spinner="Reading text from PDF...")
def extract_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


# --- 5. Main UI ---
st.title("💬 Interactive PDF Chat Agent")
st.write("Upload a PDF in the sidebar and ask questions about it.")
st.markdown("---")

if not API_KEY:
    st.error(
        "GEMINI_API_KEY is not set. Add it in Streamlit Cloud → App settings → "
        "Secrets (or in `.streamlit/secrets.toml` when running locally)."
    )
    st.stop()

if uploaded_file is None:
    st.info("👈 Please upload a PDF file in the left sidebar to start chatting.")
    st.stop()

document_context = extract_text(uploaded_file.getvalue())
if not document_context.strip():
    st.warning(
        "No text could be extracted. This PDF may be a scanned image — "
        "this app only reads PDFs that contain selectable text."
    )
    st.stop()

st.sidebar.success(f"Processed: {uploaded_file.name}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask something about this PDF..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Agent is analyzing the document..."):
            try:
                client = genai.Client(api_key=API_KEY)
                prompt = f"""You are a helpful research assistant.
Answer the question using only the context extracted from the PDF below.
If the answer cannot be found in the text, say 'I cannot find that in the document'.

Context:
{document_context}

Question: {user_query}
Answer:"""
                answer = ask_gemini(client, MODEL, prompt) or "I could not generate an answer."
            except Exception as e:
                answer = None
                st.error(f"Error calling AI engine: {e}")

        if answer:
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
