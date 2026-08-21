const BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export type Book = {
  id: string;
  title: string;
  author: string;
  total_words: number;
  total_pages: number;
  n_chunks: number;
  license: string;
  words_per_page: number;
  // null until the reader declares their own edition; when set, total_pages
  // is their count and words_per_page is derived from it.
  edition_pages: number | null;
};

export type Session = {
  id: string;
  book_id: string;
  position: number;
  page: number;
};

export type Evidence = {
  chunk_id: string;
  chapter: number;
  page: number;
  word_start: number;
  word_end: number;
  // "neighbour" is a chunk widened in around a hit rather than matched itself
  // (firewall._with_neighbours), so it reads as context, not evidence.
  via: "vector" | "keyword" | "both" | "neighbour";
  similarity: number;
  text: string;
};

export type Candidate = {
  book_id: string;
  title: string;
  author: string;
  fmt: string;
  indexed: boolean;
};

export type Job = {
  id: string;
  query: string;
  status: "queued" | "running" | "done" | "failed";
  book_id: string | null;
  message: string;
};

export type Outcome = "answered" | "refused_spoiler" | "refused_thin";

export type ChatReply = {
  answer: string;
  outcome: Outcome;
  intent: string | null;
  position: number;
  n_eligible: number;
  n_withheld: number;
  evidence: Evidence[];
};

// The status matters to one caller: a 404 on a session id is recoverable
// (the backend restarted and lost the row), where every other failure is
// something to show the reader. A bare Error cannot express that.
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* body was not json; keep the status text */
    }
    throw new ApiError(detail, res.status);
  }
  // 204 has no body at all, and res.json() would throw on the empty string
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

type Health = {
  status: string;
  books_indexed: number;
  answer_provider: string;
};

// Where the backend's live provider key came from. `hint` is the last four
// characters and nothing more -- the key itself never comes back over the
// wire, so the UI can say WHICH key is in use without ever holding it again.
export type ApiKeyState = {
  provider: string;
  // none: this provider needs no key · missing: it needs one and has none
  // environment: the deployment's own key · override: one pasted from here
  source: "none" | "missing" | "environment" | "override";
  hint: string;
};

export const api = {
  health: () => req<Health>("/health"),

  // A free-tier host sleeps after ~15 minutes idle, so the first request
  // after a quiet spell hits a container that is still booting -- and while
  // it boots it also hydrates the index from Atlas and loads the embedding
  // model. That surfaces as a connection failure, not a 503, so a single
  // fetch would report "cannot reach the API" for a backend that is merely
  // waking. Retry to a deadline and let the caller say so.
  wake: async (onWaiting?: () => void): Promise<Health> => {
    const deadline = Date.now() + 90_000;
    for (let attempt = 1; ; attempt++) {
      try {
        return await req<Health>("/health");
      } catch (e) {
        if (Date.now() > deadline) throw e;
        onWaiting?.();
        // back off, but cap it: a cold start is usually 30-60s
        await new Promise((r) => setTimeout(r, Math.min(1500 * attempt, 5000)));
      }
    }
  },

  books: () => req<Book[]>("/books"),

  // search permitted sources (Project Gutenberg via Gutendex). Does not
  // download -- that is startIngest.
  resolve: (query: string) =>
    req<Candidate[]>("/books/resolve", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),

  // returns 202 with a job id; indexing runs for minutes on the server
  startIngest: (query: string) =>
    req<Job>("/books/ingest", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),

  jobStatus: (job_id: string) => req<Job>(`/books/ingest/${job_id}`),

  // a book the reader already owns. Gutenberg carries only public-domain
  // work, so anything still in copyright can only arrive this way.
  upload: async (
    file: File,
    title: string,
    author: string,
    pages?: number | null,
  ) => {
    const form = new FormData();
    form.append("file", file);
    form.append("title", title);
    form.append("author", author);
    if (pages) form.append("pages", String(pages));
    const res = await fetch(BASE + "/books/upload", {
      method: "POST",
      body: form, // no Content-Type: the browser sets the multipart boundary
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        detail = (await res.json()).detail ?? detail;
      } catch {
        /* not json */
      }
      throw new Error(detail);
    }
    return (await res.json()) as Job;
  },

  // the reader's own edition. null returns the book to our derived paging.
  setEdition: (book_id: string, pages: number | null) =>
    req<Book>(`/books/${book_id}/edition`, {
      method: "PATCH",
      body: JSON.stringify({ pages }),
    }),

  // removes the chunks, the FTS rows, the embedding matrix, the stored
  // source file and any session transcripts. There is no undo.
  deleteBook: (book_id: string) =>
    req<void>(`/books/${book_id}`, { method: "DELETE" }),

  createSession: (book_id: string, position = 0) =>
    req<Session>("/sessions", {
      method: "POST",
      body: JSON.stringify({ book_id, position }),
    }),

  setPage: (session_id: string, page: number) =>
    req<Session>(`/sessions/${session_id}/position`, {
      method: "PATCH",
      body: JSON.stringify({ page }),
    }),

  // The provider key the server is answering with. Reported, never echoed.
  apiKeyState: () => req<ApiKeyState>("/config/api-key"),

  // Paste your own key. The server verifies it against the provider before
  // installing it, so a rejected key leaves the working one in place and
  // this throws with the provider's reason.
  setApiKey: (api_key: string) =>
    req<ApiKeyState>("/config/api-key", {
      method: "POST",
      body: JSON.stringify({ api_key }),
    }),

  // Drop the pasted key and fall back to whatever the deployment was
  // configured with.
  clearApiKey: () => req<ApiKeyState>("/config/api-key", { method: "DELETE" }),

  chat: (session_id: string, question: string, position: number) =>
    req<ChatReply>("/chat", {
      method: "POST",
      body: JSON.stringify({ session_id, question, position }),
    }),
};
