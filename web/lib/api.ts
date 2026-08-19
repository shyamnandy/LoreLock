const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type Book = {
  id: string;
  title: string;
  author: string;
  total_words: number;
  total_pages: number;
  n_chunks: number;
  license: string;
  words_per_page: number;
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
  via: "vector" | "keyword" | "both";
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
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () =>
    req<{ status: string; books_indexed: number; answer_provider: string }>(
      "/health",
    ),

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
  upload: async (file: File, title: string, author: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("title", title);
    form.append("author", author);
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

  chat: (session_id: string, question: string, position: number) =>
    req<ChatReply>("/chat", {
      method: "POST",
      body: JSON.stringify({ session_id, question, position }),
    }),
};
