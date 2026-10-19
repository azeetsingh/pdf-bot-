import io
import os
import time

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader


# --- 1. Page config ---
st.set_page_config(page_title="PDF Chat AI Agent", page_icon="🤖", layout="wide")

# Backup models, tried in order if the main one is overloaded or unavailable
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-2.5-flash"]
MAX_PDF_CHARS = 40000       # smaller prompt = faster answers
CALL_TIMEOUT_MS = 30000     # one request may take at most 30 seconds
TOTAL_TIMEOUT_S = 60        # give up after 60 seconds in total

# Words that suggest the question needs live web information
SEARCH_HINTS = (
    "latest", "news", "today", "current", "now", "price", "weather", "score",
    "stock", "who is", "who won", "trending", "recent", "release", "update",
    "2025", "2026", "search", "google", "online", "web",
)


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


# --- 3. Fast streaming answer: text appears as soon as Gemini starts writing ---
def needs_search(query: str, always: bool) -> bool:
    if always:
        return True
    q = query.lower()
    return any(hint in q for hint in SEARCH_HINTS)


def stream_answer(client, model, prompt, use_search, sources_out, status):
    """Yields text pieces as they arrive. Falls back to other models if a model
    fails before sending any text. Fills sources_out with web sources."""
    deadline = time.time() + TOTAL_TIMEOUT_S
    models = [model] + [m for m in FALLBACK_MODELS if m != model]
    errors = []

    # Search only on the main model; backups answer without search (faster, safer)
    plans = []
    if use_search:
        plans.append((models[0], True))
    plans.extend((m, False) for m in models)

    for m, with_search in plans:
        if time.time() > deadline:
            errors.append("Stopped: took too long")
            break
        status(f"Asking {m}" + (" with web search..." if with_search else "..."))
        config = None
        if with_search:
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        started = False
        try:
            last_metadata = None
            for chunk in client.models.generate_content_stream(
                model=m, contents=prompt, config=config
            ):
                try:
                    md = chunk.candidates[0].grounding_metadata
                    if md:
                        last_metadata = md
                except Exception:
                    pass
                if chunk.text:
                    started = True
                    yield chunk.text

            if last_metadata:
                for c in (last_metadata.grounding_chunks or []):
                    web = getattr(c, "web", None)
                    if web and web.uri and all(web.uri != s["url"] for s in sources_out):
                        sources_out.append({"title": web.title or web.uri, "url": web.uri})
            if started:
                return
            errors.append(f"[{m}] empty reply")
        except Exception as e:
            errors.append(f"[{m}] {str(e)[:160]}")
            if started:
                return  # part of the answer is already shown; keep it
            continue

    raise RuntimeError(" | ".join(errors[-4:]))


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
always_search = st.sidebar.checkbox(
    "🌐 Always use web search (slower)",
    value=False,
    help="Off: Olivia searches the web only when your question needs live information.",
)

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
- Greetings and small talk ("hi", "hello", "how are you", "thanks"): reply warmly and briefly, \
and say what you can do (chat about anything and answer questions about an uploaded PDF).
- General questions: answer them helpfully and clearly, even if no PDF is uploaded.
- When your answer relies on web results, say so briefly.
- Questions about the uploaded document: answer using the PDF context below. If the answer is \
not in the document, say you could not find it in the document, and then offer whatever general \
help you can.
- If the user asks about "the PDF" or "the document" but no document is provided, kindly ask \
them to upload a PDF using the sidebar.
- Keep answers clear and concise. Never leave the user without a reply."""


def build_prompt(user_query: str) -> str:
    # Include the last few turns so follow-up questions make sense
    history = ""
    for m in st.session_state.messages[-6:]:
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
        status_box = st.empty()  # shows which model is being asked
        try:
            client = genai.Client(
                api_key=API_KEY,
                http_options=types.HttpOptions(timeout=CALL_TIMEOUT_MS),
            )
            full_prompt = build_prompt(user_query)
            use_search = needs_search(user_query, always_search)
            answer = st.write_stream(
                stream_answer(
                    client, MODEL, full_prompt, use_search, sources,
                    status=lambda text: status_box.caption(text),
                )
            )
        except Exception as e:
            st.error(
                "Google's AI service is busy or returned an error. "
                f"Please try again in a minute.\n\nDetails: {e}"
            )
        status_box.empty()

        if answer:
            show_sources(sources)

            # Save both turns to history (only once)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
