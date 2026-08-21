import { useEffect, useMemo, useRef, useState } from "react";

/* The landing page.

   It is deliberately its own screen, and the app proper is only reached by
   asking for it. A reader who arrives while the free-tier backend is still
   booting should land on something that explains the idea and lets them play
   with it -- not on a half-empty reading UI waiting on a fetch. So the boot
   state is reported here, in a caption under the button, and nothing
   navigates on its own. */

export type BootStatus = "connecting" | "waking" | "ready" | "offline";

type Props = {
  status: BootStatus;
  /** how many books the backend already has indexed */
  count: number;
  error: string | null;
  provider: string;
  /** the gate: open the reading UI, or the add dialog if the shelf is bare */
  onEnter: (e: React.MouseEvent<HTMLButtonElement>) => void;
  onUpload: (e: React.MouseEvent<HTMLButtonElement>) => void;
  /** retry the boot handshake after it gave up */
  onRetry: () => void;
  /** the shelf, rendered by App so select and delete stay in one place */
  library: React.ReactNode;
};

/* ------------------------------------------------------------------ */
/* the demo                                                            */

/* One book, hard-coded, and called a demo everywhere it appears. It exists
   to make the mechanic tangible before the reader has indexed anything:
   drag the boundary, watch the searchable prefix grow, ask one of two
   questions and see one of them refuse because its evidence is still ahead
   of you. Nothing here touches the API, which is the point -- it works
   while the server is still waking. */
const DEMO = {
  title: "Frankenstein",
  author: "Mary Shelley",
  pages: 280,
  cells: 96,
};

type Ask = {
  q: string;
  /** the page its evidence first appears on; before that it must refuse */
  needs: number;
  cite: string;
  answer: string;
};

const ASKS: Ask[] = [
  {
    q: "Why does Victor leave Geneva?",
    needs: 41,
    cite: "p. 38 · chunk 0034",
    answer:
      "To study natural philosophy at Ingolstadt. The pages behind you set it up as his father’s plan, and a death in the family delays the departure.",
  },
  {
    q: "Does the creature ever get a companion?",
    needs: 174,
    cite: "p. 171 · chunk 0149",
    answer:
      "It is promised one: Victor agrees to build her, and the passages you have read describe him starting the work.",
  },
];

function Demo() {
  const [page, setPage] = useState(12);
  const [ask, setAsk] = useState(0);
  // The opening sweep runs once and yields the moment the reader touches
  // anything -- a control that keeps moving under the hand is a broken one.
  const swept = useRef(false);

  useEffect(() => {
    if (swept.current) return;
    const reduced = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (reduced) {
      setPage(96);
      swept.current = true;
      return;
    }
    const id = setInterval(() => {
      if (swept.current) {
        clearInterval(id);
        return;
      }
      setPage((p) => {
        if (p >= 96) {
          swept.current = true;
          clearInterval(id);
          return 96;
        }
        return p + 3;
      });
    }, 22);
    return () => clearInterval(id);
  }, []);

  const grab = (n: number) => {
    swept.current = true;
    setPage(n);
  };

  const pct = (page / DEMO.pages) * 100;
  const eligible = Math.max(1, Math.round((page / DEMO.pages) * DEMO.cells));
  const sealed = DEMO.cells - eligible;
  const current = ASKS[ask];
  const answered = page >= current.needs;

  const cells = useMemo(
    () => Array.from({ length: DEMO.cells }, (_, i) => i),
    [],
  );

  return (
    <section className="card card-cream demo" aria-label="Interactive demo">
      <h2 className="card-title">
        The boundary
        <span className="count">demo</span>
      </h2>

      <div className="demo-book">
        <strong>{DEMO.title}</strong>
        <span>
          {DEMO.author} · {DEMO.pages} pages
        </span>
      </div>

      {/* The passages, in narrative order. Filled cells are the prefix that
          is actually in the searched matrix; the rest are not dimmed copies
          of themselves -- they are simply not there. */}
      <div
        className="chunks"
        role="img"
        aria-label={`${eligible} of ${DEMO.cells} passages inside the boundary`}
      >
        {cells.map((i) => (
          <span key={i} className={i < eligible ? "chunk on" : "chunk"} />
        ))}
      </div>

      <label className="demo-drag" htmlFor="demo-page">
        <span>Drag to the page you are on</span>
        <b>
          {page}
          <small>/{DEMO.pages}</small>
        </b>
      </label>
      <input
        id="demo-page"
        className="track"
        type="range"
        min={1}
        max={DEMO.pages}
        value={page}
        style={{ ["--pct" as string]: `${pct}%` }}
        onChange={(e) => grab(Number(e.target.value))}
        onPointerDown={() => {
          swept.current = true;
        }}
      />

      <div className="demo-counts">
        <span className="tally ok">
          <b>{eligible}</b> searchable
        </span>
        <span className="tally no">
          <b>{sealed}</b> sealed
        </span>
      </div>

      <div className="demo-asks" role="group" aria-label="Sample questions">
        {ASKS.map((a, i) => (
          <button
            key={a.q}
            type="button"
            className={`chip${i === ask ? " on" : ""}`}
            aria-pressed={i === ask}
            onClick={() => {
              swept.current = true;
              setAsk(i);
            }}
          >
            {a.q}
          </button>
        ))}
      </div>

      <div className={`demo-reply ${answered ? "answered" : "withheld"}`}>
        <span className="tag-mini">
          {answered
            ? "answered from what you have read"
            : "withheld · would spoil it"}
        </span>
        <p>
          {answered
            ? current.answer
            : `Nothing behind page ${page} settles this. The passages that do sit ${
                current.needs - page
              } pages ahead, and they are not in the slice being searched.`}
        </p>
        {answered && <p className="cite-mini">{current.cite}</p>}
      </div>

      <p className="demo-foot">
        A canned illustration, not a live model — but the boundary is the real
        mechanic: search only ever runs over the filled prefix.
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* step icons -- drawn, so they inherit currentColor like the rest      */

function IconSearch() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 18 18"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="8" cy="8" r="5.1" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M11.8 11.8 15.5 15.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

function IconSlider() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 18 18"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M2.2 9h13.6"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="11" cy="9" r="2.9" fill="currentColor" />
    </svg>
  );
}

function IconQuote() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 18 18"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M3.2 3.8h11.6M3.2 7.4h11.6M3.2 11h7.4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <path
        d="M3.2 14.6h4.2"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        opacity="0.4"
      />
    </svg>
  );
}

/* ------------------------------------------------------------------ */

export default function Landing({
  status,
  count,
  error,
  provider,
  onEnter,
  onUpload,
  onRetry,
  library,
}: Props) {
  const waking = status === "waking" || status === "connecting";

  const badge =
    status === "offline"
      ? "Server unreachable"
      : waking
        ? "Waking the server"
        : count
          ? `${count} ${count === 1 ? "book" : "books"} indexed`
          : "Nothing indexed yet";

  const label =
    status === "offline"
      ? "Try again"
      : waking
        ? "Waking the server…"
        : count
          ? "Start reading"
          : "Add your first book";

  /* The one line that has to carry the cold start. It sits under the button
     the reader is about to press and says why it is not ready yet -- small,
     because a sleeping container is ordinary, not an incident. */
  const caption =
    status === "offline" ? (
      <>
        Could not reach the API. It may still be asleep — press <b>Try again</b>
        , or come back in a minute.
      </>
    ) : status === "waking" ? (
      <>
        Free tier: the server sleeps when idle, so the first load can take up to
        a minute. The demo works while you wait.
      </>
    ) : status === "connecting" ? (
      <>Checking the server…</>
    ) : count ? (
      <>
        Ready. {count} {count === 1 ? "book is" : "books are"} indexed, and every
        answer cites the pages it used.
      </>
    ) : (
      <>
        Free public-domain books from Project Gutenberg — or a DRM-free file you
        already own.
      </>
    );

  return (
    <>
      <header className="hero hero-landing">
        <div className="hero-copy">
          <span
            className={`badge${waking ? " badge-wait" : ""}${
              status === "offline" ? " badge-down" : ""
            }`}
          >
            <i />
            {badge}
          </span>

          <h1>
            Ask your book anything.
            <br />
            <em>Spoil nothing.</em>
          </h1>

          <p className="lede">
            Tell BookMate how far you have read. It answers from the pages
            behind you and nothing else — because the pages ahead are not in the
            array it searches.
          </p>

          <div className="hero-cta">
            <button
              className="btn-gold"
              onClick={status === "offline" ? onRetry : onEnter}
              disabled={waking}
            >
              {waking && <span className="spin spin-ink" aria-hidden="true" />}
              {label}
            </button>
            <button className="btn-quiet" onClick={onUpload} disabled={waking}>
              Upload one you own
            </button>
          </div>

          <p className={`hero-note${waking ? " is-wait" : ""}`} role="status">
            {waking && <span className="waking-dot" aria-hidden="true" />}
            {caption}
          </p>

          <ul className="hero-facts">
            <li>No account, no tracking</li>
            <li>Answers quote their sources</li>
            <li>Your page number is the only setting</li>
          </ul>
        </div>

        <Demo />
      </header>

      {error && status !== "offline" && <div className="error">{error}</div>}

      <section className="how">
        <h2 className="how-title">Three moves</h2>
        <ol className="steps">
          <li className="step">
            <span className="step-i">
              <IconSearch />
            </span>
            <b>01 · Add</b>
            <p>
              Search Project Gutenberg, or upload the DRM-free copy you already
              own. It is chunked in narrative order and embedded once.
            </p>
          </li>
          <li className="step">
            <span className="step-i">
              <IconSlider />
            </span>
            <b>02 · Mark your page</b>
            <p>
              Drag the slider to where you actually are. That number is the
              whole boundary — there is no second setting to get wrong.
            </p>
          </li>
          <li className="step">
            <span className="step-i">
              <IconQuote />
            </span>
            <b>03 · Ask</b>
            <p>
              Characters, motives, what you forgot two hundred pages ago. Every
              passage the model saw is listed, with the page it came from.
            </p>
          </li>
        </ol>
      </section>

      {count > 0 && <div className="landing-shelf">{library}</div>}

      <p className="footnote">
        The spoiler boundary is not a prompt instruction. Chunks are stored in
        narrative order, so the eligible set is a contiguous prefix of an array
        and the searched matrix is the slice <code>E[:n_p]</code> — future
        passages are absent from the operand, not filtered out of it.
        {provider ? ` Answer model: ${provider}.` : ""}
      </p>
    </>
  );
}
