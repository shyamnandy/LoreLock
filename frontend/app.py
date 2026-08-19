"""Streamlit UI.

The position slider is the whole demo: drag it forward and evidence that was
withheld becomes available, with no change to the model or the prompt
template. The evidence panel makes the firewall visible -- it reports what
was admitted and what was withheld on every turn.
"""
from __future__ import annotations

import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app.agents import llm  # noqa: E402
from app.config import WORDS_PER_PAGE  # noqa: E402
from app.graph import ask  # noqa: E402
from app.ingestion.pipeline import ingest_from_gutenberg  # noqa: E402
from app.store import db, vectors  # noqa: E402

st.set_page_config(page_title="Spoiler-Safe Reading Companion",
                   page_icon="\U0001f512", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
if "history" not in st.session_state:
    st.session_state.history = []

conn = db.connect()
books = [b for b in db.list_books(conn) if vectors.exists(b.id)]

with st.sidebar:
    st.header("Library")
    if books:
        labels = {f"{b.title} — {b.author}": b for b in books}
        choice = st.selectbox("Book", list(labels))
        book = labels[choice]
    else:
        book = None
        st.info("No books indexed yet. Add one below.")

    st.divider()
    st.caption("Add a public-domain book")
    query = st.text_input("Title or author", placeholder="Dracula")
    if st.button("Fetch and index", disabled=not query):
        with st.spinner("Resolving, downloading, chunking, embedding…"):
            try:
                added = ingest_from_gutenberg(query)
                st.success(f"Indexed {added.title}")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not index that: {exc}")

    st.divider()
    st.caption(f"Answer model: `{llm.provider_name()}`")

if book is None:
    st.title("Spoiler-Safe Reading Companion")
    st.write("Index a book from the sidebar to begin.")
    st.stop()

st.title(book.title)
st.caption(f"{book.author} · {book.total_words:,} words · {book.total_pages} pages "
           f"· {book.n_chunks} chunks · {book.license}")

page = st.slider("How far have you read?", min_value=0, max_value=book.total_pages,
                 value=min(40, book.total_pages), step=1,
                 help="Pages are this edition's, at 250 words per page.")
position = page * WORDS_PER_PAGE

left, right = st.columns([3, 2], gap="large")

with left:
    question = st.text_input("Ask about the book",
                             placeholder="Who is this character and why did he leave?")
    if st.button("Ask", type="primary", disabled=not question):
        with st.spinner("Retrieving within your position…"):
            out = ask(st.session_state.session_id, book.id, question, position)
        st.session_state.history.insert(0, (page, question, out))

    for pg, q, out in st.session_state.history:
        icon = {"answered": "\u2705", "refused_spoiler": "\U0001f512",
                "refused_thin": "\U0001f50d"}.get(out["outcome"], "\u2753")
        with st.container(border=True):
            st.markdown(f"**{q}**  \n<span style='opacity:.6'>asked at page {pg}"
                        f" · {icon} {out['outcome'].replace('_', ' ')}</span>",
                        unsafe_allow_html=True)
            st.write(out["answer"])

with right:
    st.subheader("Firewall")
    if st.session_state.history:
        out = st.session_state.history[0][2]
        n_p = out.get("n_p", 0)
        withheld = out.get("n_withheld", 0)
        c1, c2 = st.columns(2)
        c1.metric("Chunks eligible", n_p)
        c2.metric("Chunks withheld", withheld)
        st.progress(n_p / max(1, n_p + withheld))
        st.caption(f"Intent classified as **{out.get('intent', '—')}**. "
                   f"The model received {len(out.get('evidence', []))} passages.")

        st.subheader("Evidence admitted")
        for e in out.get("evidence", []):
            with st.expander(
                f"ch {e.chunk.chapter} · p{e.chunk.page} · {e.via} · {e.score:.4f}"
            ):
                st.write(e.chunk.text)
                st.caption(f"`{e.chunk.id}` words {e.chunk.word_start}–{e.chunk.word_end}")
    else:
        st.caption("Ask something to see what the firewall admitted.")
