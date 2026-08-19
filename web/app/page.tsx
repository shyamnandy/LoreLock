"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type Book,
  type Candidate,
  type ChatReply,
  type Job,
  type Session,
} from "@/lib/api";

type Turn = { page: number; question: string; reply: ChatReply };

const OUTCOME_LABEL: Record<ChatReply["outcome"], string> = {
  answered: "answered from the pages you've read",
  refused_spoiler: "withheld — answering would spoil it",
  refused_thin: "not enough evidence yet",
};

export default function Page() {
  const [books, setBooks] = useState<Book[]>([]);
  const [book, setBook] = useState<Book | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [page, setPage] = useState(40);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState("");
  const [error, setError] = useState<string | null>(null);

  // adding a book
  const [search, setSearch] = useState("");
  const [candidates, setCandidates] = useState<Candidate[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [upTitle, setUpTitle] = useState("");
  const [upAuthor, setUpAuthor] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadLibrary = useCallback(async () => {
    const [list, health] = await Promise.all([api.books(), api.health()]);
    setBooks(list);
    setProvider(health.answer_provider);
    return list;
  }, []);

  useEffect(() => {
    loadLibrary()
      .then((list) => {
        if (list.length) setBook(list[0]);
      })
      .catch((e) =>
        setError(
          `Cannot reach the API at ${
            process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"
          }. Start the backend, or set NEXT_PUBLIC_API_URL. (${(e as Error).message})`,
        ),
      );
  }, [loadLibrary]);

  // a new book means a new session; clear the transcript with it
  useEffect(() => {
    if (!book) return;
    setTurns([]);
    const start = Math.min(40, book.total_pages);
    setPage(start);
    api
      .createSession(book.id, start * book.words_per_page)
      .then(setSession)
      .catch((e) => setError((e as Error).message));
  }, [book]);

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

  const ingesting = job && (job.status === "queued" || job.status === "running");

  const libraryPanel = (
    <div className="card">
      <h2>Your library</h2>

      {books.length > 0 && (
        <div style={{ marginBottom: "1.1rem" }}>
          <label htmlFor="book">Reading</label>
          <select
            id="book"
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
        </div>
      )}

      <label htmlFor="search">Add a book you&rsquo;re reading</label>
      <div className="ask-row" style={{ marginTop: 0 }}>
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
          onClick={runSearch}
          disabled={searching || !!ingesting || search.trim().length < 2}
        >
          {searching ? "Searching…" : "Search"}
        </button>
      </div>
      <p className="page-sub" style={{ marginTop: "0.5rem" }}>
        Public-domain titles from Project Gutenberg.
      </p>

      <div className="upload-block">
        <label htmlFor="upfile">Or upload a book you own</label>
        <input
          id="upfile"
          ref={fileRef}
          type="file"
          accept=".epub,.txt,.html,.htm,.xhtml"
          disabled={!!ingesting || uploading}
        />
        <div className="upload-meta">
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
          onClick={uploadBook}
          disabled={!!ingesting || uploading}
          style={{ marginTop: "0.6rem", width: "100%" }}
        >
          {uploading ? "Uploading…" : "Upload and index"}
        </button>
        <p className="page-sub" style={{ marginTop: "0.5rem" }}>
          EPUB, TXT or HTML, up to 25 MB. Your own DRM-free copy — nothing is
          shared, and title and author are read from the file if you leave
          them blank.
        </p>
      </div>

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
              ? "Downloading, chunking and embedding. This takes a few minutes; you can leave the page open."
              : job.message}
          </p>
          {ingesting && <div className="indeterminate" />}
        </div>
      )}

      {candidates && !ingesting && (
        <div className="candidates">
          {candidates.length === 0 && (
            <p className="hint">
              Nothing permitted found for that. Try the exact title, or the
              author&rsquo;s name.
            </p>
          )}
          {candidates.map((c) => (
            <div key={c.book_id} className="candidate">
              <div>
                <div className="cand-title">{c.title}</div>
                <div className="cand-author">
                  {c.author} · {c.fmt}
                </div>
              </div>
              <button
                onClick={() => addBook(c)}
                disabled={c.indexed}
                title={c.indexed ? "Already in your library" : "Download and index"}
              >
                {c.indexed ? "In library" : "Add"}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <main className="wrap">
      <header className="masthead">
        <p className="eyebrow">Temporal RAG · spoiler firewall</p>
        <h1>{book ? book.title : "Spoiler-Safe Reading Companion"}</h1>
        <p className="byline">
          {book
            ? `${book.author} · ${book.total_words.toLocaleString()} words · ${book.total_pages} pages · ${book.n_chunks} chunks · ${book.license}`
            : "Ask your book anything without spoiling what comes next."}
        </p>
      </header>

      {error && <div className="error">{error}</div>}

      {books.length === 0 && !error ? (
        <div style={{ maxWidth: "34rem" }}>
          <p className="hint" style={{ marginBottom: "1.2rem" }}>
            Nothing indexed yet. Search for the book you&rsquo;re reading and it
            will be downloaded from Project Gutenberg and indexed with
            narrative-position metadata.
          </p>
          {libraryPanel}
        </div>
      ) : (
        <div className="columns">
          <div>
            <div className="card">
              <h2>Reading position</h2>
              <div className="slider-head">
                <label htmlFor="pos">How far have you read?</label>
                <span className="page-readout">
                  p{page}
                  <span className="page-sub"> / {book?.total_pages ?? 0}</span>
                </span>
              </div>
              <input
                id="pos"
                type="range"
                min={0}
                max={book?.total_pages ?? 100}
                value={page}
                onChange={(e) => setPage(Number(e.target.value))}
                disabled={!book}
              />
              <p className="page-sub">
                word offset {position.toLocaleString()} · pages are this
                edition&rsquo;s, at {book?.words_per_page ?? 250} words each
              </p>

              <div className="ask-row">
                <input
                  type="text"
                  placeholder="Who is this character, and why did he leave?"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && ask()}
                  disabled={!session || busy}
                />
                <button
                  onClick={ask}
                  disabled={!session || busy || !question.trim()}
                >
                  {busy ? "Retrieving…" : "Ask"}
                </button>
              </div>
            </div>

            {turns.length === 0 && (
              <div className="card">
                <p className="hint">
                  Ask something, then drag the slider forward and ask the same
                  question again. Nothing about the model changes between the
                  two — only the slice of the book it is allowed to see.
                </p>
              </div>
            )}

            {turns.map((t, i) => (
              <article key={i} className={`turn ${t.reply.outcome}`}>
                <p className="turn-q">{t.question}</p>
                <p className="turn-meta">
                  asked at page {t.page} · {OUTCOME_LABEL[t.reply.outcome]}
                  {t.reply.intent ? ` · intent: ${t.reply.intent}` : ""}
                </p>
                <p className="turn-a">{t.reply.answer}</p>
              </article>
            ))}
          </div>

          <aside>
            <div className="card">
              <h2>Firewall</h2>
              {latest ? (
                <>
                  <div className="meters">
                    <div className="meter eligible">
                      <div className="meter-n">{latest.n_eligible}</div>
                      <div className="meter-label">chunks eligible</div>
                    </div>
                    <div className="meter withheld">
                      <div className="meter-n">{latest.n_withheld}</div>
                      <div className="meter-label">chunks withheld</div>
                    </div>
                  </div>
                  <div
                    className="bar"
                    role="img"
                    aria-label={`${latest.n_eligible} of ${total} chunks eligible`}
                  >
                    <div
                      className="bar-eligible"
                      style={{ width: `${eligiblePct}%` }}
                    />
                    <div
                      className="bar-withheld"
                      style={{ width: `${100 - eligiblePct}%` }}
                    />
                  </div>
                  <p className="page-sub" style={{ marginTop: "0.7rem" }}>
                    the model received {latest.evidence.length} passages, none
                    past word {latest.position.toLocaleString()}
                  </p>
                </>
              ) : (
                <p className="hint">
                  Ask something to see what the firewall admitted.
                </p>
              )}
            </div>

            {latest && latest.evidence.length > 0 && (
              <div className="card">
                <h2>Evidence admitted</h2>
                {latest.evidence.map((e) => (
                  <details key={e.chunk_id}>
                    <summary>
                      <span>
                        ch {e.chapter} · p{e.page}
                      </span>
                      <span>
                        <span className={`via ${e.via}`}>{e.via}</span>{" "}
                        {e.similarity.toFixed(3)}
                      </span>
                    </summary>
                    <p>{e.text}</p>
                    <p className="cite">
                      {e.chunk_id} · words {e.word_start}–{e.word_end}
                    </p>
                  </details>
                ))}
              </div>
            )}

            {libraryPanel}
          </aside>
        </div>
      )}

      <p className="footnote">
        The spoiler boundary is not a prompt instruction. Chunks are stored in
        narrative order, so the eligible set is a contiguous prefix of an array
        and the searched matrix is the slice <code>E[:n_p]</code> — future
        passages are absent from the operand, not filtered out of it.
        {provider ? ` Answer model: ${provider}.` : ""}
      </p>
    </main>
  );
}
