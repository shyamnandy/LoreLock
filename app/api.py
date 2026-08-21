"""FastAPI backend.

Deployed separately from the frontend because it needs a real disk: the
chunk store, the FTS index and the embedding matrices are files, and
ingestion writes to them for minutes at a time. Vercel serves the UI and
calls this.

The endpoints mirror the design's API table. Positions are always word
offsets on the wire; pages are a presentation concern the client owns.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from contextlib import asynccontextmanager, closing

from fastapi import (BackgroundTasks, FastAPI, File, Form, HTTPException,
                     UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents import llm
# Aliased: line 479 binds `app = api` for the ASGI entrypoint
# convention, which shadows the `app` PACKAGE inside this module --
# so a plain `import app.config` would resolve to the FastAPI object
# at call time. Imported as a module, not as a value, because the
# ingest flag must stay overridable after import.
from app import config as app_config
from app.config import RAW, WORDS_PER_PAGE
from app.graph import ask
from app.ingestion import sources, upload
from app.ingestion.pipeline import ingest_file, ingest_from_gutenberg
from app.store import db, embedders, seed, vectors

@asynccontextmanager
async def _lifespan(_: FastAPI):
    # Before the first connect, which would create an empty database at
    # DB_PATH and so convince hydrate() that this host already has one.
    if installed := seed.hydrate():
        print(f"hydrated the index from seed/ ({installed} book(s))")
    # Before the first question, and before anything reports which key is in
    # use: a key pasted in an earlier run is only persistent if it is back in
    # the environment by the time a backend reads it.
    if restored := llm.restore_saved_keys():
        print(f"restored a saved API key for {', '.join(restored)}")
    with closing(db.connect()) as conn:
        # a background ingest belongs to one process; if we are starting up, any
        # job still marked in-flight was owned by a process that is now gone
        if reaped := db.reap_orphan_jobs(conn):
            print(f"reaped {reaped} orphaned ingest job(s)")
    yield


api = FastAPI(title="Spoiler-Safe Reading Companion", version="0.1.0",
              lifespan=_lifespan)

# the Vercel deployment origin, plus local dev
_origins = [o for o in os.environ.get("CORS_ORIGINS", "").split(",") if o] or [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
api.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)


# -- schemas ---------------------------------------------------------------

class ApiKeyIn(BaseModel):
    # A key is a credential, not free text: bound so a paste of the wrong
    # thing (a whole .env file, a URL) is rejected before it reaches the
    # provider. `repr=False` keeps it out of pydantic's __repr__, which is
    # what a validation error or an unhandled traceback would print.
    api_key: str = Field(min_length=8, max_length=512, repr=False)


class ApiKeyOut(BaseModel):
    """Where the live key came from. Never the key itself -- see llm.key_hint."""
    provider: str
    source: str          # none | missing | environment | override
    hint: str            # last four characters, "" when there is no key


class BookOut(BaseModel):
    id: str
    title: str
    author: str
    total_words: int
    total_pages: int
    n_chunks: int
    license: str
    words_per_page: int = WORDS_PER_PAGE
    # None until the reader tells us their edition; the client shows the
    # difference between our paging and theirs when it is set.
    edition_pages: int | None = None


class ResolveIn(BaseModel):
    query: str = Field(min_length=2, max_length=200)


class CandidateOut(BaseModel):
    book_id: str
    title: str
    author: str
    fmt: str
    indexed: bool


class JobOut(BaseModel):
    id: str
    query: str
    status: str          # queued | running | done | failed
    book_id: str | None = None
    message: str = ""


class SessionIn(BaseModel):
    book_id: str
    position: int = 0


class SessionOut(BaseModel):
    id: str
    book_id: str
    position: int
    page: int


class EditionIn(BaseModel):
    # None clears the declaration and returns the book to our own paging
    pages: int | None = Field(default=None, ge=1, le=20_000)


class PositionIn(BaseModel):
    position: int | None = None
    page: int | None = None


class ChatIn(BaseModel):
    session_id: str
    question: str = Field(min_length=2, max_length=500)
    position: int | None = None


class EvidenceOut(BaseModel):
    chunk_id: str
    chapter: int
    page: int
    word_start: int
    word_end: int
    via: str
    similarity: float
    text: str


class ChatOut(BaseModel):
    answer: str
    outcome: str
    intent: str | None = None
    position: int
    n_eligible: int
    n_withheld: int
    evidence: list[EvidenceOut]


# -- helpers ---------------------------------------------------------------

def _book_out(b) -> BookOut:
    return BookOut(id=b.id, title=b.title, author=b.author,
                   total_words=b.total_words, total_pages=b.total_pages,
                   n_chunks=b.n_chunks, license=b.license,
                   words_per_page=b.words_per_page,
                   edition_pages=b.edition_pages)


def _page_of(conn, book_id: str, position: int) -> int:
    """Word offset -> the page number the reader would recognise."""
    b = db.get_book(conn, book_id)
    return position // (b.words_per_page if b else WORDS_PER_PAGE)


# -- routes ----------------------------------------------------------------

@api.get("/health")
def health() -> dict:
    """Also reports which build is serving.

    Without this there is no way to tell a deployed fix from an undeployed
    one except by guessing at behaviour, and behaviour is a bad proxy: a
    restart clears in-process caches, so the same probe can read either way.
    RENDER_GIT_COMMIT is injected by the platform; "unknown" locally.
    """
    with closing(db.connect()) as conn:
        ready = [b for b in db.list_books(conn) if vectors.exists(b.id)]
    return {"status": "ok", "books_indexed": len(ready),
            "answer_provider": llm.provider_name(),
            "embed_model": embedders.model_name(),
            "commit": os.environ.get("RENDER_GIT_COMMIT", "unknown")[:7]}


# -- the provider key ------------------------------------------------------
#
# Lets a reader supply their OWN key from the UI instead of running on the
# deployment's. The free tiers this runs on meter tokens per day per account,
# so the shared key is the resource that actually runs out; pasting a personal
# one is the difference between a demo that answers and one that 429s.
#
# The override is the process environment, which is what app.agents.llm reads
# and therefore the only place a change takes effect without a redeploy. Two
# consequences worth being clear about rather than hiding:
#   - it is PROCESS-WIDE. There is one environment, so a pasted key serves
#     every concurrent reader, not just the one who pasted it.
#   - it lives in memory only. A restart -- which a free instance does every
#     time it sleeps -- returns the process to its configured key.
# Both are why DELETE restores the baseline rather than blanking it: falling
# back to the deployment's own key is the safe resting state.

@api.get("/config/api-key", response_model=ApiKeyOut)
def get_api_key() -> ApiKeyOut:
    return ApiKeyOut(**llm.key_state())


@api.post("/config/api-key", response_model=ApiKeyOut)
def set_api_key(body: ApiKeyIn) -> ApiKeyOut:
    """Verify a pasted key, then make it the live one.

    Verified BEFORE it is installed, and the current key is left untouched if
    the check fails. Installing first would let one typo take down a working
    deployment until someone redeployed it.
    """
    try:
        llm.verify_key(body.api_key)
        return ApiKeyOut(**llm.set_api_key(body.api_key))
    except llm.LLMError as exc:
        raise HTTPException(400, str(exc)) from exc


@api.delete("/config/api-key", response_model=ApiKeyOut)
def clear_api_key() -> ApiKeyOut:
    """Discard the override and fall back to the configured key."""
    try:
        return ApiKeyOut(**llm.clear_api_key())
    except llm.LLMError as exc:
        raise HTTPException(400, str(exc)) from exc


@api.get("/books", response_model=list[BookOut])
def list_books() -> list[BookOut]:
    with closing(db.connect()) as conn:
        return [_book_out(b) for b in db.list_books(conn) if vectors.exists(b.id)]


@api.get("/books/{book_id}", response_model=BookOut)
def get_book(book_id: str) -> BookOut:
    with closing(db.connect()) as conn:
        book = db.get_book(conn, book_id)
    if book is None or not vectors.exists(book_id):
        raise HTTPException(404, "book not indexed")
    return _book_out(book)


@api.patch("/books/{book_id}/edition", response_model=BookOut)
def set_edition(book_id: str, body: EditionIn) -> BookOut:
    """Declare how many pages the reader's own edition has.

    Our page numbers come from total_words // 250, which no printed edition
    agrees with. Given the real count we rescale words-per-page instead, so
    page N of their copy is the same fraction of the book either way. The
    text and the chunks do not move, so a session in progress keeps its exact
    word position and is simply relabelled.
    """
    with closing(db.connect()) as conn:
        if db.get_book(conn, book_id) is None:
            raise HTTPException(404, "no such book")
        try:
            b = db.set_edition_pages(conn, book_id, body.pages)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    return _book_out(b)


@api.delete("/books/{book_id}", status_code=204)
def delete_book(book_id: str) -> None:
    """Remove a book from the library, with the derived state it owns.

    Three stores hold something for a book and all three must go, or the
    next ingest of the same title inherits it: the rows (chunks, the FTS
    index, sessions and their transcripts), the embedding matrix, and the
    downloaded source file.

    The matrix is dropped first. A book row without a matrix is inert --
    /books/{id} already 404s on exactly that condition -- whereas a matrix
    without rows is a file the next same-id ingest would find and trust.
    """
    with closing(db.connect()) as conn:
        if db.get_book(conn, book_id) is None:
            raise HTTPException(404, "no such book")
        vectors.delete(book_id)
        db.delete_book(conn, book_id)

    # uploads keep their source as RAW/{book_id}{suffix}; the suffix varies
    # (.epub, .txt, .html) so match on the stem
    for stored in RAW.glob(f"{book_id}.*"):
        stored.unlink(missing_ok=True)


@api.post("/books/resolve", response_model=list[CandidateOut])
def resolve(body: ResolveIn) -> list[CandidateOut]:
    """Search permitted sources. Does not download -- that is /books/ingest."""
    try:
        found = sources.search(body.query, limit=5)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"source lookup failed: {exc}") from exc
    with closing(db.connect()) as conn:
        return [
            CandidateOut(
                book_id=c.book_id, title=c.title, author=c.author, fmt=c.fmt,
                indexed=(db.get_book(conn, c.book_id) is not None
                         and vectors.exists(c.book_id)),
            )
            for c in found
        ]


def _require_ingest_allowed() -> None:
    """Refuse an ingest this host cannot afford to finish.

    Checked in the REQUEST, not in the background worker: the whole point is
    to fail before a job row exists and before a file is written, so the
    client gets a synchronous error instead of a job that polls its way to
    a failure -- or, on a killed container, never resolves at all.
    """
    if not app_config.ALLOW_RUNTIME_INGEST:
        raise HTTPException(
            503,
            "this deployment serves a prebuilt library and cannot index new "
            "books -- indexing needs more CPU and memory than the instance "
            "has. Index locally, run scripts/build_seed.py, and redeploy.",
        )


def _run_ingest(job_id: str, query: str,
                edition_pages: int | None = None) -> None:
    """Background worker. Downloads, chunks and embeds -- minutes of work.

    `edition_pages` mirrors _run_upload_ingest: without it the body below
    raises NameError, but only on the success path, so a five-minute embed
    would complete and then report the job failed.
    """
    with closing(db.connect()) as conn:
        db.update_job(conn, job_id, "running", "resolving source")
    try:
        book = ingest_from_gutenberg(query)
    except LookupError as exc:
        with closing(db.connect()) as conn:
            db.update_job(conn, job_id, "failed", str(exc))
    except Exception as exc:  # noqa: BLE001
        with closing(db.connect()) as conn:
            db.update_job(conn, job_id, "failed", f"{type(exc).__name__}: {exc}")
    else:
        with closing(db.connect()) as conn:
            if edition_pages:
                db.set_edition_pages(conn, book.id, edition_pages)
            db.update_job(conn, job_id, "done",
                          f"indexed {book.n_chunks} chunks", book.id)


@api.post("/books/ingest", response_model=JobOut, status_code=202)
def ingest(body: ResolveIn, background: BackgroundTasks) -> JobOut:
    """Start an ingest and return immediately.

    Downloading, chunking and embedding a novel takes minutes -- far longer
    than any browser or proxy will hold a connection open. The client polls
    /books/ingest/{job_id}.
    """
    _require_ingest_allowed()
    query = body.query.strip()
    with closing(db.connect()) as conn:
        if existing := db.active_job_for(conn, query):
            return JobOut(**existing)          # don't start a second one
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = db.create_job(conn, job_id, query)

    background.add_task(_run_ingest, job_id, query)
    return JobOut(**job)


def _run_upload_ingest(job_id: str, path: str, book_id: str,
                       title: str, author: str,
                       edition_pages: int | None = None) -> None:
    """Background worker for an uploaded file."""
    with closing(db.connect()) as conn:
        db.update_job(conn, job_id, "running", "parsing and indexing")
    try:
        book = ingest_file(Path(path), book_id=book_id, title=title,
                           author=author, source="upload", source_url="",
                           license="user-supplied copy")
    except Exception as exc:  # noqa: BLE001
        with closing(db.connect()) as conn:
            db.update_job(conn, job_id, "failed", f"{type(exc).__name__}: {exc}")
    else:
        with closing(db.connect()) as conn:
            if edition_pages:
                db.set_edition_pages(conn, book.id, edition_pages)
            db.update_job(conn, job_id, "done",
                          f"indexed {book.n_chunks} chunks", book.id)


@api.post("/books/upload", response_model=JobOut, status_code=202)
async def upload_book(background: BackgroundTasks,
                      file: UploadFile = File(...),
                      title: str = Form(""),
                      author: str = Form(""),
                      pages: int | None = Form(None)) -> JobOut:
    """Index a book the reader already owns.

    Gutenberg only carries public-domain work, so anything still in copyright
    can only arrive this way -- from the reader's own DRM-free file.
    """
    _require_ingest_allowed()
    data = await file.read()
    try:
        path, book_id, _digest = upload.store(file.filename or "book.txt", data)
    except upload.UploadRejected as exc:
        raise HTTPException(422, str(exc)) from exc

    # prefer what the reader typed, then the file's own metadata, then the name
    meta_title, meta_author = (upload.epub_metadata(path)
                               if path.suffix == ".epub" else (None, None))
    title = title.strip() or meta_title or Path(file.filename or "").stem or "Untitled"
    author = author.strip() or meta_author or "Unknown"

    with closing(db.connect()) as conn:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = db.create_job(conn, job_id, f"upload: {title}")

    background.add_task(_run_upload_ingest, job_id, str(path), book_id,
                        title, author, pages if pages and pages > 0 else None)
    return JobOut(**job)


@api.get("/books/ingest/{job_id}", response_model=JobOut)
def ingest_status(job_id: str) -> JobOut:
    with closing(db.connect()) as conn:
        job = db.get_job(conn, job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return JobOut(**job)


@api.post("/sessions", response_model=SessionOut)
def create_session(body: SessionIn) -> SessionOut:
    with closing(db.connect()) as conn:
        if db.get_book(conn, body.book_id) is None:
            raise HTTPException(404, "book not indexed")
        sid = f"s-{uuid.uuid4().hex[:12]}"
        s = db.create_session(conn, sid, body.book_id, body.position)
        page = _page_of(conn, s["book_id"], s["position"])
    return SessionOut(**s, page=page)


@api.get("/sessions/{session_id}", response_model=SessionOut)
def read_session(session_id: str) -> SessionOut:
    with closing(db.connect()) as conn:
        s = db.get_session(conn, session_id)
        page = _page_of(conn, s["book_id"], s["position"]) if s else 0
    if s is None:
        raise HTTPException(404, "no such session")
    return SessionOut(**s, page=page)


@api.patch("/sessions/{session_id}/position", response_model=SessionOut)
def update_position(session_id: str, body: PositionIn) -> SessionOut:
    if body.position is None and body.page is None:
        raise HTTPException(422, "supply either position (words) or page")
    with closing(db.connect()) as conn:
        existing = db.get_session(conn, session_id)
        if existing is None:
            raise HTTPException(404, "no such session")
        if body.position is not None:
            position = body.position
        else:
            book = db.get_book(conn, existing["book_id"])
            wpp = book.words_per_page if book else WORDS_PER_PAGE
            position = body.page * wpp
        try:
            s = db.set_position(conn, session_id, position)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        page = _page_of(conn, s["book_id"], s["position"]) if s else 0
    if s is None:
        raise HTTPException(404, "no such session")
    return SessionOut(**s, page=page)


@api.get("/sessions/{session_id}/history")
def history(session_id: str) -> list[dict]:
    with closing(db.connect()) as conn:
        if db.get_session(conn, session_id) is None:
            raise HTTPException(404, "no such session")
        return db.session_history(conn, session_id)


@api.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    with closing(db.connect()) as conn:
        session = db.get_session(conn, body.session_id)
        if session is None:
            raise HTTPException(404, "no such session")
        # the client may move the slider and ask in one gesture; persist the
        # new position first so the session stays the single source of truth
        if body.position is not None and body.position != session["position"]:
            session = db.set_position(conn, body.session_id, body.position)

    out = ask(body.session_id, session["book_id"], body.question,
              session["position"])

    # Evidence is labelled in the same pages as the slider. Chunk.page uses the
    # 250-word default because a chunk does not know which book it belongs to;
    # here we do, so a declared edition relabels the citations too.
    with closing(db.connect()) as conn:
        b = db.get_book(conn, session["book_id"])
    wpp = b.words_per_page if b else WORDS_PER_PAGE

    return ChatOut(
        answer=out["answer"],
        outcome=out["outcome"],
        intent=out.get("intent"),
        position=session["position"],
        n_eligible=out.get("n_p", 0),
        n_withheld=out.get("n_withheld", 0),
        evidence=[
            EvidenceOut(
                chunk_id=e.chunk.id, chapter=e.chunk.chapter,
                page=e.chunk.word_start // wpp + 1,
                word_start=e.chunk.word_start, word_end=e.chunk.word_end,
                via=e.via, similarity=round(e.similarity, 4), text=e.chunk.text,
            )
            for e in out.get("evidence", [])
        ],
    )


app = api  # uvicorn / ASGI entrypoint convention
