"""LangGraph orchestration.



The topology is not decorative. The spoiler problem produces genuine

structure that a linear chain cannot express:



    two conditional branches   assess_evidence -> answer | oracle

                               oracle -> refuse_spoiler | refuse_thin

    one cycle                  validate -> answer, bounded by retry_count

    one typed state            `position` written once at entry, read-only



The firewall is a node the graph must pass through, and it is deliberately

not an agent.

"""

from __future__ import annotations



import hashlib

from contextlib import closing

from typing import Literal, TypedDict



from langgraph.graph import END, START, StateGraph



from app.agents import analyzer, llm, validator

from app.agents.prompts import (ANSWER_SYSTEM, build_answer_prompt,

                                fit_to_budget)

from app.config import (MAX_EVIDENCE_CHARS, MAX_VALIDATION_RETRIES,

                        SUFFICIENCY_FLOOR, TOP_K)

from app.models import Evidence, Intent, Outcome

from app.retrieval.firewall import SafeRetriever

from app.retrieval.oracle import answered_later

from app.store import db



REFUSAL_SPOILER = (

    "I can't answer that from what you've read so far without giving away "

    "something that comes later. Ask me again once you're further in."

)

REFUSAL_THIN = (

    "I don't have enough to go on from the pages you've read yet. If it was "

    "only touched on in passing, it may not be spelled out until later."

)





class ReaderState(TypedDict, total=False):

    # inputs -- position is written once, here, and never recomputed

    session_id: str

    book_id: str

    question: str

    position: int



    # analysis

    intent: Intent



    # retrieval (always post-firewall)

    evidence: list[Evidence]

    n_p: int

    n_withheld: int

    evidence_sufficient: bool

    top_similarity: float

    answered_later: bool | None



    # generation

    draft: str

    citations: list[str]

    retry_count: int

    validation_reason: str

    provider_error: str



    # output

    answer: str

    outcome: Outcome  # answered | refused_spoiler | refused_thin | error

    prompt_text: str
    shown_evidence: list[Evidence]





# -- nodes -----------------------------------------------------------------



def analyze_question(state: ReaderState) -> dict:

    return {"intent": analyzer.classify(state["question"]),

            "retry_count": state.get("retry_count", 0)}





def retrieve(state: ReaderState) -> dict:

    with closing(db.connect()) as conn:

        retriever = SafeRetriever(conn, state["book_id"], state["position"])

        evidence = retriever.search(state["question"], k=TOP_K)

        return {"evidence": evidence, "n_p": retriever.n_p,

                "n_withheld": retriever.n_withheld()}





def assess_evidence(state: ReaderState) -> dict:

    """Judge on raw cosine, never on the RRF score.



    RRF encodes rank only -- its top hit scores the same whether the passage

    is on point or entirely unrelated, so thresholding it would let any

    non-empty result set through and the spoiler branch would be unreachable.

    """

    evidence = state.get("evidence", [])

    best = max((e.similarity for e in evidence), default=0.0)

    return {"evidence_sufficient": bool(evidence) and best >= SUFFICIENCY_FLOOR,

            "top_similarity": best}





def spoiler_oracle(state: ReaderState) -> dict:

    """Air-gapped: reads STORE B, returns one boolean, never any text."""

    return {"answered_later": answered_later(

        state["book_id"], state["n_p"], state["question"])}





def answer(state: ReaderState) -> dict:

    evidence = fit_to_budget(state["evidence"], MAX_EVIDENCE_CHARS)

    prompt = build_answer_prompt(state["question"], evidence)

    if reason := state.get("validation_reason"):

        prompt += (f"\n\nYour previous draft was rejected because it {reason}. "

                   "Cite only the passages above, or reply INSUFFICIENT.")

    try:

        draft = llm.chat(ANSWER_SYSTEM, prompt, role="answer", temperature=0.0).text

    except llm.LLMError as exc:

        # NOT a refusal. Returning INSUFFICIENT here once made a rate-limited

        # provider look like correct spoiler-safe behaviour, and the eval

        # scored it as 100% refusal accuracy. Surface it as its own outcome.

        return {"draft": "", "prompt_text": prompt,

                "provider_error": str(exc)[:300]}

    return {"draft": draft, "prompt_text": prompt, "provider_error": ""}





def validate(state: ReaderState) -> dict:

    allowed = [e.chunk.id for e in state["evidence"]]

    result = validator.check(

        state["draft"], allowed,

        evidence_text="\n".join(f"[{e.chunk.id}] {e.chunk.text[:400]}"

                                for e in state["evidence"]),

    )

    return {"citations": result.cited, "validation_reason": result.reason,

            "retry_count": state.get("retry_count", 0) + (0 if result.grounded else 1)}





def surface_error(state: ReaderState) -> dict:

    return {"answer": "I could not reach the answering model just now. This is "

                      "a service problem, not a spoiler. Try again in a moment.",

            "outcome": "error"}





def refuse_spoiler(_: ReaderState) -> dict:

    return {"answer": REFUSAL_SPOILER, "outcome": "refused_spoiler"}





def refuse_thin(_: ReaderState) -> dict:

    return {"answer": REFUSAL_THIN, "outcome": "refused_thin"}





def finalise(state: ReaderState) -> dict:

    return {"answer": state["draft"], "outcome": "answered"}





# -- conditional edges -----------------------------------------------------



def route_after_assess(state: ReaderState) -> Literal["answer", "spoiler_oracle"]:

    return "answer" if state["evidence_sufficient"] else "spoiler_oracle"





def route_after_oracle(state: ReaderState) -> Literal["refuse_spoiler", "refuse_thin"]:

    return "refuse_spoiler" if state.get("answered_later") else "refuse_thin"





def route_after_validate(state: ReaderState

                         ) -> Literal["finalise", "answer", "spoiler_oracle",

                                      "surface_error"]:

    """The answer agent is the real sufficiency judge.



    It has read the passages; a cosine threshold has not. When it declines,

    the oracle decides which refusal the reader gets -- "not yet" or "not

    here" -- and that is the only place the future is consulted.

    """

    if state.get("provider_error"):

        return "surface_error"        # never dress a broken call as a refusal

    draft = state.get("draft", "").strip().upper()

    if draft.startswith("INSUFFICIENT"):

        return "spoiler_oracle"

    if state["validation_reason"] == "grounded":

        return "finalise"

    if state.get("retry_count", 0) >= MAX_VALIDATION_RETRIES:

        return "spoiler_oracle"

    return "answer"





# -- assembly --------------------------------------------------------------



def build_graph():

    g = StateGraph(ReaderState)



    g.add_node("analyze_question", analyze_question)

    g.add_node("retrieve", retrieve)

    g.add_node("assess_evidence", assess_evidence)

    g.add_node("spoiler_oracle", spoiler_oracle)

    g.add_node("answer", answer)

    g.add_node("validate", validate)

    g.add_node("finalise", finalise)

    g.add_node("refuse_spoiler", refuse_spoiler)

    g.add_node("refuse_thin", refuse_thin)

    g.add_node("surface_error", surface_error)



    g.add_edge(START, "analyze_question")

    g.add_edge("analyze_question", "retrieve")

    g.add_edge("retrieve", "assess_evidence")

    g.add_conditional_edges("assess_evidence", route_after_assess,

                            {"answer": "answer", "spoiler_oracle": "spoiler_oracle"})

    g.add_conditional_edges("spoiler_oracle", route_after_oracle,

                            {"refuse_spoiler": "refuse_spoiler",

                             "refuse_thin": "refuse_thin"})

    g.add_edge("answer", "validate")

    g.add_conditional_edges("validate", route_after_validate,

                            {"finalise": "finalise", "answer": "answer",

                             "spoiler_oracle": "spoiler_oracle",

                             "surface_error": "surface_error"})

    g.add_edge("finalise", END)

    g.add_edge("refuse_spoiler", END)

    g.add_edge("refuse_thin", END)

    g.add_edge("surface_error", END)



    return g.compile()





GRAPH = build_graph()





def ask(session_id: str, book_id: str, question: str, position: int) -> ReaderState:

    """One turn. `position` is a word offset and is authoritative."""

    final: ReaderState = GRAPH.invoke({

        "session_id": session_id, "book_id": book_id,

        "question": question, "position": position, "retry_count": 0,

    })

    _log(final)

    return final





def _log(state: ReaderState) -> None:

    prompt = state.get("prompt_text", "")

    with closing(db.connect()) as conn:

        conn.execute(

            """INSERT INTO conversations (session_id, position, question,

                   answer, outcome, evidence_ids, prompt_hash, prompt_text)

               VALUES (?,?,?,?,?,?,?,?)""",

            (state.get("session_id", ""), state.get("position", 0),

             state.get("question", ""), state.get("answer", ""),

             state.get("outcome", "refused_thin"),

             ",".join(e.chunk.id for e in state.get("evidence", [])),

             hashlib.sha256(prompt.encode()).hexdigest(), prompt),

        )

        conn.commit()

