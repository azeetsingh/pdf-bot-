import io
import os
import time

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader


# --- 1. Page config ---
st.set_page_config(page_title="PDF Chat AI Agent", page_icon="🤖", layout="wide")

# Models tried in order when the main one is overloaded (503) or rate limited (429)
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-2.5-flash"]
MAX_PDF_CHARS = 60000  # keeps prompts small so calls stay fast


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


# --- 3. Gemini calls with retry + fallback ---
def is_temporary_error(e: Exception) -> bool:
    msg = str(e)
    return any(code in msg for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "504", "DEADLINE"))


def generate_with_retry(client, model, prompt, use_search=False):
    """Try the main model, then fallback models. Retries temporary errors."""
    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    config = None
    if use_search:
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )

    errors = []
    for m in models_to_try:
        for attempt in range(3):
            try:
                return client.models.generate_content(
                    model=m, contents=prompt, config=config
                )
            except Exception as e:
                errors.append(f"[{m}] {str(e)[:200]}")
                if is_temporary_error(e):
                    time.sleep(2 * (attempt + 1))  # wait 2s, 4s, 6s
                    continue
                break  # other error (e.g. 404 bad model name): go to next model
    # Show every model's error, not just the last one
    raise RuntimeError(" | ".join(errors[-4:]))


def ask_agent(client, model, prompt):
    """Gemini with Google Search. If search fails, answer without it.
    Returns (answer_text, sources)."""
    try:
        response = generate_with_retry(client, model, prompt, use_search=True)
    except Exception:
        response = generate_with_retry(client, model, prompt, use_search=False)

    sources = []
    try:
        metadata = response.candidates[0].grounding_metadata
        for chunk in (metadata.grounding_chunks or []):
            web = getattr(chunk, "web", None)
            if web and web.uri and all(web.uri != s["url"] for s in sources):
                sources.append({"title": web.title or web.uri, "url": web.uri})
    except Exception:
        pass

    return response.text, sources


def show_sources(sources):
    if sources:
        with st.expander(f"🔎 Web sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['url']})")


API_KEY = get_api_key()
MODEL = get_model_name()

# --- 4. Sidebar ---
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


# --- 5. PDF text extraction ---
@st.cache_data(show_spinner="Reading text from PDF...")
def extract_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


# --- 6. Main UI ---
st.title("💬 Olivia — AI Agent")
st.write(
    "Chat with Olivia about anything. She can search the web for live information, "
    "and answer questions about a PDF you upload in the sidebar."
)
st.markdown("---")

if not API_KEY:
    st.error(
        "GEMINI_API_KEY is not set. Add it in Streamlit Cloud → App settings → "
        "Secrets (or in `.streamlit/secrets.toml` when running locally)."
    )
    st.stop()

# PDF is optional: the bot works as a normal assistant without it.
document_context = ""
if uploaded_file is not None:
    document_context = extract_text(uploaded_file.getvalue())
    if document_context.strip():
        st.sidebar.success(f"Processed: {uploaded_file.name}")
        if len(document_context) > MAX_PDF_CHARS:
            st.sidebar.info(
                f"Large PDF: only the first {MAX_PDF_CHARS:,} characters are used."
            )
            document_context = document_context[:MAX_PDF_CHARS]
    else:
        st.sidebar.warning(
            "No text could be extracted from this PDF (it may be a scanned image). "
            "I can still chat, but I can't read this document."
        )

SYSTEM_INSTRUCTIONS = """You are Olivia, a friendly and capable AI agent inside a chat app.
You can search the web for current information and you can read an uploaded PDF.

How to behave:
- Introduce yourself as Olivia when greeting someone for the first time.
- Use web search on your own whenever a question needs up-to-date or external facts \
(news, prices, recent events, current information, anything you are unsure about). \
Do not search for greetings, small talk, or questions fully answered by the PDF.
- When your answer relies on web results, say so briefly.
- Greetings and small talk ("hi", "hello", "how are you", "thanks"): reply warmly and naturally, \
like a helpful assistant, and briefly say what you can do (chat about anything and answer \
questions about an uploaded PDF).
- General questions: answer them helpfully and clearly, even if no PDF is uploaded.
- Questions about the uploaded document: answer using the PDF context below. If the answer is \
not in the document, say you could not find it in the document, and then offer whatever general \
help you can.
- If the user asks about "the PDF" or "the document" but no document is provided, kindly ask \
them to upload a PDF using the sidebar.
- Keep answers clear and reasonably concise. Never leave the user without a reply."""


def build_prompt(user_query: str) -> str:
    # Include the last few turns so follow-up questions make sense
    history = ""
    for m in st.session_state.messages[-8:]:
        who = "User" if m["role"] == "user" else "Assistant"
        history += f"{who}: {m['content']}\n"

    if document_context.strip():
        doc_block = f"PDF context (from '{uploaded_file.name}'):\n{document_context}"
    else:
        doc_block = "PDF context: (no document uploaded)"

    return f"""{SYSTEM_INSTRUCTIONS}

{doc_block}

Conversation so far:
{history}
User: {user_query}
Assistant:"""


if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hi! I'm **Olivia**, your AI agent. Ask me anything, I can search the "
            "web for live info, or upload a PDF in the sidebar and I'll answer "
            "questions about it."
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources"))

placeholder = (
    "Ask something about your PDF or anything else..."
    if document_context.strip()
    else "Say hi or ask me anything..."
)

if user_query := st.chat_input(placeholder):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        answer = None
        sources = []
        with st.spinner("Olivia is thinking..."):
            try:
                client = genai.Client(
                    api_key=API_KEY,
                    http_options=types.HttpOptions(timeout=60000),  # 60 seconds
                )
                full_prompt = build_prompt(user_query)
                answer, sources = ask_agent(client, MODEL, full_prompt)
                if not answer:
                    answer = "Sorry, I couldn't generate a reply. Please try again."
            except Exception as e:
                st.error(
                    "Google's AI service is busy or returned an error. "
                    f"Please try again in a minute.\n\nDetails: {e}"
                )

        if answer:
            st.markdown(answer)
            show_sources(sources)

            # Save both turns to history (only once)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
