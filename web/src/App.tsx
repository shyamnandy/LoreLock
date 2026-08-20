import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type Book,
  type Candidate,
  type ChatReply,
  type Job,
  type Session,
} from "@/api";

type Turn = { page: number; question: string; reply: ChatReply };

const OUTCOME_TAG: Record<ChatReply["outcome"], string> = {
  answered: "answered from what you've read",
  refused_spoiler: "withheld · would spoil it",
  refused_thin: "not enough evidence yet",
};

/* An open book whose left page is filled and right page is only outlined:
   the read prefix and the pages ahead, which is the whole idea. */
function Mark({ size = 32 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
    >
      {/* pages behind you -- solid */}
      <path
        d="M16 10.4c-2.4-1.9-6-2.8-9.6-2.5a1.3 1.3 0 0 0-1.2 1.3v12.9c0 .8.6 1.4 1.4 1.3 3.4-.2 6.9.7 9.4 2.6z"
        fill="currentColor"
      />
      {/* pages ahead -- absent, not hidden */}
      <path
        d="M16 10.4c2.4-1.9 6-2.8 9.6-2.5a1.3 1.3 0 0 1 1.2 1.3v12.9c0 .8-.6 1.4-1.4 1.3-3.4-.2-6.9.7-9.4 2.6z"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
        opacity="0.45"
      />
    </svg>
  );
}

/* Trash: lid, body, two tines. Drawn rather than an emoji so it inherits
   currentColor and stays aligned with the text beside it. */
function Trash({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M2.75 4.25h10.5M6.25 4.25V3a.75.75 0 0 1 .75-.75h2a.75.75 0 0 1 .75.75v1.25"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
      <path
        d="M3.9 4.25h8.2l-.62 8.3a1 1 0 0 1-1 .95H5.52a1 1 0 0 1-1-.95l-.62-8.3Z"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
      <path
        d="M6.6 6.9v3.9M9.4 6.9v3.9"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
    </svg>
  );
}

function Close({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M4 4l8 8M12 4l-8 8"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default function App() {
  const [books, setBooks] = useState<Book[]>([]);
  const [book, setBook] = useState<Book | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [page, setPage] = useState(40);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState("");
  const [error, setError] = useState<string | null>(null);
  // true only while a sleeping backend is being woken; see api.wake
  const [waking, setWaking] = useState(false);
  // the book id currently being deleted, so its row alone shows the wait
  const [deleting, setDeleting] = useState<string | null>(null);

  // adding a book
  const [search, setSearch] = useState("");
  const [candidates, setCandidates] = useState<Candidate[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [upTitle, setUpTitle] = useState("");
  const [upAuthor, setUpAuthor] = useState("");
  const [uploading, setUploading] = useState(false);
  // The library is a drawer, not a permanent column. With a book open it sits
  // shut so the thread and the firewall own the screen; the nav picker already
  // covers the only thing it was doing there full time, which is switching book.
  const [libOpen, setLibOpen] = useState(true);
  // Adding a book is an errand, not a place: it happens in a dialog opened
  // from the header, so the shelf below is only ever a list of what you have.
  const [modal, setModal] = useState<null | "search" | "upload">(null);
  // declaring the edition's page count
  const [editionOpen, setEditionOpen] = useState(false);
  const [editionInput, setEditionInput] = useState("");
  const [savingEdition, setSavingEdition] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const openerRef = useRef<HTMLElement | null>(null);

  const loadLibrary = useCallback(async () => {
    const [list, health] = await Promise.all([api.books(), api.health()]);
    setBooks(list);
    setProvider(health.answer_provider);
    return list;
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // The API may be a sleeping free-tier container. Wait it out rather
        // than reporting an outage for a backend that is only booting.
        await api.wake(() => !cancelled && setWaking(true));
        if (cancelled) return;
        const list = await loadLibrary();
        if (cancelled) return;
        if (list.length) setBook(list[0]);
      } catch (e) {
        if (cancelled) return;
        setError(
          `Cannot reach the API at ${
            import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"
          }. Start the backend, or set VITE_API_URL. (${(e as Error).message})`,
        );
      } finally {
        if (!cancelled) setWaking(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loadLibrary]);

  // A new *book* means a new session; clear the transcript with it. Keyed on
  // the id, not the object: re-declaring the edition replaces the object
  // in place, and that must not throw away the conversation.
  useEffect(() => {
    if (!book) {
      setLibOpen(true);
      return;
    }
    setLibOpen(false);
    setTurns([]);
    const start = Math.min(40, book.total_pages);
    setPage(start);
    api
      .createSession(book.id, start * book.words_per_page)
      .then(setSession)
      .catch((e) => setError((e as Error).message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [book?.id]);

  // poll an in-flight ingest
  useEffect(() => {
    if (!job || job.status === "done" || job.status === "failed") return;
    pollRef.current = setInterval(async () => {
      try {
        const next = await api.jobStatus(job.id);
        setJob(next);
        if (next.status === "done") {
          const list = await loadLibrary();
          const added = list.find((b) => b.id === next.book_id);
          if (added) {
            setBook(added);
            setCandidates(null);
            setSearch("");
            // the errand is done; the reader is now looking at the book
            setModal(null);
          }
        }
      } catch {
        /* transient; the next tick retries */
      }
    }, 3000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [job, loadLibrary]);

  const runSearch = useCallback(async () => {
    if (search.trim().length < 2) return;
    setSearching(true);
    setError(null);
    try {
      setCandidates(await api.resolve(search.trim()));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSearching(false);
    }
  }, [search]);

  const uploadBook = useCallback(async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      setJob(await api.upload(file, upTitle, upAuthor));
      setUpTitle("");
      setUpAuthor("");
      if (fileRef.current) fileRef.current.value = "";
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }, [upTitle, upAuthor]);

  /* Declaring an edition rescales words-per-page. The reader has not moved
     -- keep the same word offset and relabel it, so the slider still points
     at the same sentence it did a moment ago. */
  const saveEdition = useCallback(
    async (pages: number | null) => {
      if (!book) return;
      setSavingEdition(true);
      setError(null);
      const wordsIn = page * book.words_per_page;
      try {
        const updated = await api.setEdition(book.id, pages);
        setBooks((list) =>
          list.map((b) => (b.id === updated.id ? updated : b)),
        );
        setBook(updated);
        setPage(
          Math.min(
            updated.total_pages,
            Math.round(wordsIn / updated.words_per_page),
          ),
        );
        setEditionOpen(false);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setSavingEdition(false);
      }
    },
    [book, page],
  );

  const addBook = useCallback(async (c: Candidate) => {
    setError(null);
    try {
      setJob(await api.startIngest(`${c.title} ${c.author}`));
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const latest = turns[0]?.reply ?? null;
  const total = latest ? latest.n_eligible + latest.n_withheld : 0;
  const eligiblePct = total ? (latest!.n_eligible / total) * 100 : 0;
  const position = useMemo(
    () => (book ? page * book.words_per_page : 0),
    [book, page],
  );
  const readPct =
    book && book.total_pages ? (page / book.total_pages) * 100 : 0;

  const ask = useCallback(async () => {
    if (!session || !question.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const reply = await api.chat(session.id, question.trim(), position);
      setTurns((t) => [{ page, question: question.trim(), reply }, ...t]);
      setQuestion("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [session, question, position, page]);

  // Deleting is not undoable and takes the transcripts with it, so it asks
  // first. The open book is only replaced if it is the one that went.
  const removeBook = useCallback(
    async (target: Book) => {
      const ok = window.confirm(
        `Remove “${target.title}” from your library?

` +
          "Its passages, embeddings and every conversation about it are " +
          "deleted. This cannot be undone.",
      );
      if (!ok) return;
      setDeleting(target.id);
      setError(null);
      try {
        await api.deleteBook(target.id);
        const list = await loadLibrary();
        setBook((cur) =>
          cur && cur.id === target.id ? (list[0] ?? null) : cur,
        );
      } catch (e) {
        setError(`Could not remove ${target.title}: ${(e as Error).message}`);
      } finally {
        setDeleting(null);
      }
    },
    [loadLibrary],
  );

  const ingesting =
    job && (job.status === "queued" || job.status === "running");

  const closeModal = useCallback(() => {
    setModal(null);
    // send focus back where it came from, so the header does not lose its
    // place for anyone driving this from the keyboard
    openerRef.current?.focus();
    openerRef.current = null;
  }, []);

  const openModal = useCallback(
    (which: "search" | "upload", e: React.MouseEvent<HTMLButtonElement>) => {
      openerRef.current = e.currentTarget;
      setError(null);
      setModal(which);
    },
    [],
  );

  useEffect(() => {
    if (!modal) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeModal();
    };
    document.addEventListener("keydown", onKey);
    // the page behind a dialog must not scroll under it
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    // focus the field the reader came here to type in
    requestAnimationFrame(() =>
      dialogRef.current
        ?.querySelector<HTMLElement>("input, button:not([disabled])")
        ?.focus(),
    );
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [modal, closeModal]);

  /* ---------------------------------------------------------------- */

  /* The shelf, and nothing else. Search and upload used to live under it,
     which turned a list of seven books into a column of three stacked forms. */
  const libraryPanel = (
    <section className="card card-cream panel" id="library">
      <button
        type="button"
        className="panel-head"
        onClick={() => setLibOpen((v) => !v)}
        aria-expanded={libOpen}
        aria-controls="library-body"
      >
        <span className="card-title">Your library</span>
        <span className="panel-meta">
          {books.length} {books.length === 1 ? "book" : "books"}
          <span className={`chev${libOpen ? " up" : ""}`} aria-hidden="true" />
        </span>
      </button>

      <div className="panel-body" id="library-body" hidden={!libOpen}>
        <div className="shelf">
          {books.map((b) => (
            <div
              key={b.id}
              className={`shelf-row${b.id === book?.id ? " is-open" : ""}`}
            >
              <button
                className="shelf-pick"
                onClick={() => setBook(b)}
                /* the row ellipses, so the tooltip carries the whole title
                   whether or not this is the book already open */
                title={`${b.title} — ${b.author}${
                  b.id === book?.id ? " (currently open)" : ""
                }`}
              >
                <span className="cand-title">{b.title}</span>
                <span className="cand-author">
                  {b.author} · {b.total_pages} pages
                </span>
              </button>
              <button
                className="icon-btn icon-btn-danger"
                onClick={() => removeBook(b)}
                disabled={deleting === b.id}
                aria-label={`Remove ${b.title} from your library`}
                title="Remove from library"
              >
                {deleting === b.id ? <span className="spin" /> : <Trash />}
              </button>
            </div>
          ))}
        </div>

        {/* An indexing run outlives the dialog it was started from, so it
            reports here as well -- the dialog can be closed and the book
            still arrives. */}
        {job && !modal && (
          <div className={`job ${job.status}`}>
            <strong>
              {job.status === "done"
                ? "Indexed"
                : job.status === "failed"
                  ? "Could not index"
                  : "Indexing"}
              {" — "}
              {job.query}
            </strong>
            <p>
              {ingesting
                ? "Downloading, chunking and embedding. This takes a few minutes; you can leave the page open."
                : job.message}
            </p>
            {ingesting && <div className="indeterminate" />}
          </div>
        )}
      </div>
    </section>
  );

  /* ---------------------------------------------------------------- */

  const addDialog = modal && (
    <div
      className="backdrop"
      onMouseDown={(e) => {
        // only a press that both starts and ends on the backdrop dismisses;
        // a drag that began inside the dialog must not close it
        if (e.target === e.currentTarget) closeModal();
      }}
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        ref={dialogRef}
      >
        <div className="modal-head">
          <h2 className="modal-title" id="modal-title">
            Add a book
          </h2>
          <button
            type="button"
            className="icon-btn"
            onClick={closeModal}
            aria-label="Close"
            title="Close"
          >
            <Close />
          </button>
        </div>

        {/* Two ways in, one dialog: the header button picks the tab you land
            on, and you can change your mind without starting over. */}
        <div className="tabs" role="tablist" aria-label="How to add a book">
          <button
            role="tab"
            aria-selected={modal === "search"}
            className={`tab${modal === "search" ? " on" : ""}`}
            onClick={() => setModal("search")}
          >
            Search Gutenberg
          </button>
          <button
            role="tab"
            aria-selected={modal === "upload"}
            className={`tab${modal === "upload" ? " on" : ""}`}
            onClick={() => setModal("upload")}
          >
            Upload your own
          </button>
        </div>

        <div className="modal-body">
          {error && <div className="error">{error}</div>}

          {modal === "search" ? (
            <>
              <label htmlFor="search">Title, or author</label>
              <div className="row">
                <input
                  id="search"
                  type="text"
                  placeholder="Frankenstein, or Mary Shelley"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && runSearch()}
                  disabled={!!ingesting}
                />
                <button
                  className="btn-dark"
                  onClick={runSearch}
                  disabled={
                    searching || !!ingesting || search.trim().length < 2
                  }
                >
                  {searching ? "Searching…" : "Search"}
                </button>
              </div>
              <p className="sub">
                {searching
                  ? "Searching Project Gutenberg — the catalogue API is slow from our host, so this can take up to a minute."
                  : "Public-domain titles from Project Gutenberg."}
              </p>

              {candidates && !ingesting && (
                <div className="results">
                  {candidates.length === 0 && (
                    <p className="hint">
                      Nothing permitted found for that. Try the exact title, or
                      the author&rsquo;s name.
                    </p>
                  )}
                  {candidates.map((c) => (
                    <div key={c.book_id} className="cand">
                      <div>
                        <div className="cand-title">{c.title}</div>
                        <div className="cand-author">
                          {c.author} · {c.fmt}
                        </div>
                      </div>
                      <button
                        className="btn-gold btn-sm"
                        onClick={() => addBook(c)}
                        disabled={c.indexed}
                        title={
                          c.indexed
                            ? "Already in your library"
                            : "Download and index"
                        }
                      >
                        {c.indexed ? "In library" : "Add"}
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <>
              <label htmlFor="upfile">Your file</label>
              <input
                id="upfile"
                ref={fileRef}
                type="file"
                accept=".epub,.txt,.html,.htm,.xhtml"
                disabled={!!ingesting || uploading}
              />
              <div className="split">
                <input
                  type="text"
                  placeholder="Title (optional)"
                  value={upTitle}
                  onChange={(e) => setUpTitle(e.target.value)}
                  disabled={!!ingesting || uploading}
                />
                <input
                  type="text"
                  placeholder="Author (optional)"
                  value={upAuthor}
                  onChange={(e) => setUpAuthor(e.target.value)}
                  disabled={!!ingesting || uploading}
                />
              </div>
              <button
                className="btn-gold btn-block"
                onClick={uploadBook}
                disabled={!!ingesting || uploading}
                style={{ marginTop: "0.7rem" }}
              >
                {uploading ? "Uploading…" : "Upload and index"}
              </button>
              <p className="sub">
                EPUB, TXT or HTML, up to 25 MB. Your own DRM-free copy — nothing
                is shared, and title and author are read from the file if you
                leave them blank.
              </p>
            </>
          )}

          {job && (
            <div className={`job ${job.status}`}>
              <strong>
                {job.status === "done"
                  ? "Indexed"
                  : job.status === "failed"
                    ? "Could not index"
                    : "Indexing"}
                {" — "}
                {job.query}
              </strong>
              <p>
                {ingesting
                  ? "Downloading, chunking and embedding. This takes a few minutes — you can close this and carry on reading."
                  : job.message}
              </p>
              {ingesting && <div className="indeterminate" />}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  /* ---------------------------------------------------------------- */

  return (
    <div className="shell">
      <nav className="nav">
        <span className="brand">
          <span className="brand-mark">
            <Mark size={21} />
          </span>
          <span className="brand-word">
            book<em>mate</em>
          </span>
        </span>

        <span className="nav-spacer" />

        <div className="nav-mid">
          {books.length > 0 && (
            <select
              className="picker"
              aria-label="Book you are reading"
              value={book?.id ?? ""}
              onChange={(e) =>
                setBook(books.find((b) => b.id === e.target.value) ?? null)
              }
            >
              {books.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.title} — {b.author}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* both routes in are tabs of the dialog this opens */}
        <button className="btn-gold" onClick={(e) => openModal("search", e)}>
          Add a book
        </button>
      </nav>

      {book ? (
        <>
          <header className="hero">
            <div>
              <span className="badge">
                <i />
                {books.length} {books.length === 1 ? "book" : "books"} indexed ·
                spoiler firewall on
              </span>
              <h1>{book.title}</h1>
              <p className="lede">
                {book.author} · {book.total_pages} pages · {book.license} —
                answers come only from the pages behind you.
              </p>

              <div className="stats">
                <div className="stat">
                  <div className="stat-n">{position.toLocaleString()}</div>
                  <div className="stat-l">words the model may read</div>
                </div>
                <div className="stat">
                  <div className="stat-n">{book.n_chunks}</div>
                  <div className="stat-l">passages indexed</div>
                </div>
                <div className="stat">
                  <div className="stat-n">{turns.length}</div>
                  <div className="stat-l">questions this session</div>
                </div>
              </div>
            </div>

            <section className="card card-cream">
              <h2 className="card-title">Reading position</h2>
              <div className="dial">
                <span className="dial-label">How far have you read?</span>
                {/* The denominator is the affordance: the number that is wrong
                    is the one you click to correct. A separate control would
                    sit above the slider forever for a once-per-book action. */}
                <span className="dial-n">
                  {page}
                  {editionOpen ? (
                    <span className="total-edit">
                      <small>/</small>
                      <input
                        type="number"
                        min={1}
                        max={20000}
                        inputMode="numeric"
                        autoFocus
                        aria-label="Total pages in your copy"
                        value={editionInput}
                        onChange={(e) => setEditionInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && Number(editionInput) > 0)
                            saveEdition(Number(editionInput));
                          if (e.key === "Escape") setEditionOpen(false);
                        }}
                        disabled={savingEdition}
                      />
                      <button
                        className="btn-gold btn-sm"
                        onClick={() => saveEdition(Number(editionInput))}
                        disabled={savingEdition || !(Number(editionInput) > 0)}
                      >
                        {savingEdition ? "…" : "Save"}
                      </button>
                      <button
                        type="button"
                        className="link"
                        onClick={() => setEditionOpen(false)}
                        disabled={savingEdition}
                      >
                        cancel
                      </button>
                    </span>
                  ) : (
                    <button
                      type="button"
                      className="total-btn"
                      title="Not your edition? Click to set your copy's real page count."
                      onClick={() => {
                        setEditionInput(String(book.total_pages));
                        setEditionOpen(true);
                      }}
                    >
                      <small>
                        {" / "}
                        {book.total_pages} <span aria-hidden="true">✎</span>
                      </small>
                    </button>
                  )}
                </span>
              </div>
              <input
                className="track"
                aria-label="How far have you read?"
                type="range"
                min={0}
                max={book.total_pages}
                value={page}
                onChange={(e) => setPage(Number(e.target.value))}
                style={
                  { ["--pct" as string]: `${readPct}%` } as React.CSSProperties
                }
              />
              <p className="sub">
                word {position.toLocaleString()} ·{" "}
                {book.edition_pages
                  ? `your edition, ${book.words_per_page} words a page`
                  : `our estimate, ${book.words_per_page} words a page`}
                {book.edition_pages && (
                  <>
                    {" · "}
                    <button
                      type="button"
                      className="link"
                      onClick={() => saveEdition(null)}
                      disabled={savingEdition}
                    >
                      use our estimate instead
                    </button>
                  </>
                )}
              </p>

              {/* Shown only while we are guessing. Once a real count is saved
                  it never comes back, so a configured book carries no clutter. */}
              {!book.edition_pages && !editionOpen && (
                <p className="nudge">
                  <strong>{book.total_pages} pages</strong> is our estimate at
                  250 words a page. If your copy says otherwise,{" "}
                  <button
                    type="button"
                    className="link"
                    onClick={() => {
                      setEditionInput(String(book.total_pages));
                      setEditionOpen(true);
                    }}
                  >
                    set its real page count
                  </button>{" "}
                  — page {page} will then mean the same place in the text as it
                  does in your hands.
                </p>
              )}

              <div className="ask">
                <input
                  type="text"
                  placeholder="Who is this character, and why did he leave?"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && ask()}
                  disabled={!session || busy}
                />
                <button
                  className="btn-gold"
                  onClick={ask}
                  disabled={!session || busy || !question.trim()}
                >
                  {busy ? "Retrieving…" : "Ask"}
                </button>
              </div>
            </section>
          </header>

          {waking && (
            <div className="waking" role="status">
              <span className="waking-dot" aria-hidden="true" />
              Waking the server. The free tier sleeps when idle, so the first
              load can take up to a minute.
            </div>
          )}
          {error && !modal && <div className="error">{error}</div>}

          <div className="grid">
            <div className="col">
              <section className="thread">
                <div className="thread-head">
                  <h2 className="thread-title">
                    {turns.length
                      ? `${turns.length} ${turns.length === 1 ? "question" : "questions"} about ${book.title}`
                      : "Nothing asked yet"}
                  </h2>
                  {turns.length > 1 && (
                    <span className="thread-note">newest first</span>
                  )}
                </div>

                {turns.length === 0 ? (
                  <p className="thread-empty">
                    Ask something, then drag the slider forward and ask the same
                    question again. Nothing about the model changes between the
                    two — only the slice of the book it is allowed to see.
                  </p>
                ) : (
                  turns.map((t, i) => (
                    <article className="turn" key={i}>
                      <div className="msg">
                        <div className="msg-head">
                          <span className="avatar you">You</span>
                          <span className="who">You</span>
                          <span className="when">at page {t.page}</span>
                        </div>
                        <p className="msg-body">{t.question}</p>
                      </div>

                      <div className="msg">
                        <div className="msg-head">
                          <span className="avatar bot">
                            <Mark size={18} />
                          </span>
                          <span className="who">BookMate</span>
                          <span className="when">
                            {t.reply.evidence.length} passage
                            {t.reply.evidence.length === 1 ? "" : "s"} admitted
                            {t.reply.intent ? ` · ${t.reply.intent}` : ""}
                          </span>
                        </div>
                        <p className="msg-body">{t.reply.answer}</p>
                        <span className={`tag ${t.reply.outcome}`}>
                          {OUTCOME_TAG[t.reply.outcome]}
                        </span>
                      </div>
                    </article>
                  ))
                )}
              </section>
            </div>

            <div className="col">
              {/* Reports on a turn, so it waits for one. The empty state was
                  a titled card carrying a single sentence -- pure chrome in
                  the column the thread has to share. */}
              {latest && (
                <section className="card card-dark">
                  <h2 className="card-title">Firewall · last turn</h2>
                  <div className="meters">
                    <div className="meter ok">
                      <div className="meter-n">{latest.n_eligible}</div>
                      <div className="meter-l">chunks eligible</div>
                    </div>
                    <div className="meter no">
                      <div className="meter-n">{latest.n_withheld}</div>
                      <div className="meter-l">chunks withheld</div>
                    </div>
                  </div>
                  <div
                    className="bar"
                    role="img"
                    aria-label={`${latest.n_eligible} of ${total} chunks eligible`}
                  >
                    <div
                      className="bar-ok"
                      style={{ width: `${eligiblePct}%` }}
                    />
                    <div
                      className="bar-no"
                      style={{ width: `${100 - eligiblePct}%` }}
                    />
                  </div>
                  <p className="sub">
                    the model received {latest.evidence.length} passages, none
                    past word {latest.position.toLocaleString()}
                  </p>
                </section>
              )}

              {latest && latest.evidence.length > 0 && (
                <section className="card card-cream">
                  <h2 className="card-title">
                    Evidence admitted
                    <span className="count">{latest.evidence.length}</span>
                  </h2>
                  {/* Capped and scrolled: a dozen passages is a normal turn,
                      and unrolled they were pushing everything else off the
                      screen. The list keeps its own scrollbar instead. */}
                  <div className="ev-list">
                    {latest.evidence.map((e) => (
                      <details className="ev" key={e.chunk_id}>
                        <summary>
                          <span>
                            ch {e.chapter} · p{e.page}
                          </span>
                          <span>
                            <span className={`via ${e.via}`}>{e.via}</span>{" "}
                            {e.similarity.toFixed(3)}
                          </span>
                        </summary>
                        <p className="ev-text">{e.text}</p>
                        <p className="cite">
                          {e.chunk_id} · words {e.word_start}–{e.word_end}
                        </p>
                      </details>
                    ))}
                  </div>
                </section>
              )}

              {libraryPanel}
            </div>
          </div>
        </>
      ) : (
        <>
          <header className="hero hero-solo">
            <div>
              <span className="badge">
                <i />
                {books.length
                  ? `${books.length} indexed`
                  : "Nothing indexed yet"}
              </span>
              <h1>Ask your book anything. Spoil nothing.</h1>
              <p className="lede">
                Tell BookMate how far you&rsquo;ve read. It answers from the
                pages behind you and nothing else — because the pages ahead are
                not in the array it searches.
              </p>
              <div className="hero-cta">
                <button
                  className="btn-gold"
                  onClick={(e) => openModal("search", e)}
                >
                  Add your first book
                </button>
                <button
                  className="btn-quiet"
                  onClick={(e) => openModal("upload", e)}
                >
                  Upload one you own
                </button>
              </div>
            </div>
          </header>

          {waking && (
            <div className="waking" role="status">
              <span className="waking-dot" aria-hidden="true" />
              Waking the server. The free tier sleeps when idle, so the first
              load can take up to a minute.
            </div>
          )}
          {error && !modal && <div className="error">{error}</div>}

          <div className="steps">
            <div className="step">
              <b>01</b>
              <p>
                Search Project Gutenberg, or upload the DRM-free copy you
                already own.
              </p>
            </div>
            <div className="step">
              <b>02</b>
              <p>
                Drag the slider to the page you&rsquo;re on. That number is the
                whole boundary.
              </p>
            </div>
            <div className="step">
              <b>03</b>
              <p>
                Ask. Every passage the model saw is listed, with the page it
                came from.
              </p>
            </div>
          </div>

          {books.length > 0 && (
            <div style={{ maxWidth: "38rem" }}>{libraryPanel}</div>
          )}
        </>
      )}

      <p className="footnote">
        The spoiler boundary is not a prompt instruction. Chunks are stored in
        narrative order, so the eligible set is a contiguous prefix of an array
        and the searched matrix is the slice <code>E[:n_p]</code> — future
        passages are absent from the operand, not filtered out of it.
        {provider ? ` Answer model: ${provider}.` : ""}
      </p>

      {addDialog}
    </div>
  );
}
