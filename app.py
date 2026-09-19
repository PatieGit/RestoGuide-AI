"""RestoGuide AI - Streamlit employee view.

A pure consumer of the query pipeline. It imports `src.query` and nothing from
`src.ingestion`, so the interface structurally cannot write to the index. All
retrieval and prompt logic lives in the pipeline; if a prompt needs changing,
it changes in src/query/prompts.py, not here.
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.config import ConfigError
from src.query import QueryError, StoreNotFoundError, answer_question
from src.query.engine import EMPTY_QUESTION_TEXT
from src.query.prompts import NO_ANSWER_TEXT
from src.query.retriever import KnowledgeRetriever

PAGE_TITLE = "RestoGuide AI"

# One-click prompts for high-frequency tasks. Each submits through exactly the
# same path as a typed question -- no shortcut around the guardrails.
QUICK_ACTIONS: dict[str, str] = {
    "Opening checklist": "What is the standard opening routine and checklist before serving customers?",
    "Closing the kitchen": "What is the standard procedure for closing the kitchen at the end of the day?",
    "Allergy procedure": "What steps must be followed when a customer informs staff of a severe food allergy?",
    "Handling a complaint": "How should staff handle a customer complaint regarding cold food or an incorrect order?",
    "Beef wet fry recipe": "What is the recipe and preparation guide for Kenyan Beef Wet Fry?",
}

# A centred conversation column that reflows: comfortable on a landscape
# laptop, full width on a phone. Quick actions sit in the sidebar, which stays
# fixed while the conversation scrolls.
CSS = """
<style>
  .block-container {
      max-width: 46rem;
      padding-top: 2.5rem;
      padding-bottom: 6rem;
  }
  /* Sidebar scrolls independently of the conversation, so the buttons never
     scroll out of reach however long the history gets. */
  section[data-testid="stSidebar"] {
      width: 17rem !important;
  }
  section[data-testid="stSidebar"] > div {
      height: 100%;
      overflow-y: auto;
  }
  section[data-testid="stSidebar"] button {
      white-space: normal;
      text-align: left;
      height: auto;
      padding-top: 0.45rem;
      padding-bottom: 0.45rem;
  }
  .rg-tagline {
      opacity: 0.75;
      margin-top: -0.6rem;
      margin-bottom: 1.2rem;
  }
  .rg-source-q { font-weight: 600; margin-bottom: 0.25rem; }

  @media (max-width: 640px) {
      .block-container {
          max-width: 100%;
          padding-top: 1.2rem;
          padding-left: 0.9rem;
          padding-right: 0.9rem;
      }
      /* On a phone the sidebar is a drawer; let it use most of the screen so
         the button labels stay readable rather than wrapping to three lines. */
      section[data-testid="stSidebar"] {
          width: 85vw !important;
      }
  }
</style>
"""


@st.cache_resource(show_spinner=False)
def get_retriever() -> KnowledgeRetriever:
    """Build the retriever once per session rather than per interaction."""
    return KnowledgeRetriever()


def render_header() -> None:
    st.title("RestoGuide AI")
    st.markdown(
        '<p class="rg-tagline">Ask about any procedure, recipe, or customer '
        "preference in the restaurant's manual.</p>",
        unsafe_allow_html=True,
    )


def render_sidebar(strategy: str | None, *, show_actions: bool = True) -> str | None:
    """Quick actions live here so they stay put while the answer area scrolls.

    Returns the question behind whichever button was pressed, if any.
    """
    pending: str | None = None

    with st.sidebar:
        if show_actions:
            st.subheader("Common questions")
            for label, question in QUICK_ACTIONS.items():
                if st.button(label, use_container_width=True, key=f"qa_{label}"):
                    pending = question
            st.divider()

        st.caption(
            "Answers come only from the restaurant's own manual. If something "
            "isn't in there, RestoGuide says so rather than guessing."
        )
        st.divider()
        st.caption("**Models**")
        st.caption(f"Answers: `{config.LLM_MODEL}`")
        st.caption(f"Search: `{config.EMBEDDING_MODEL}`")
        if strategy:
            st.caption(f"Index: `{strategy}`, top {config.TOP_K} matches")

    return pending


def render_missing_store(error: Exception) -> None:
    st.warning("The knowledge base hasn't been built yet.")
    st.markdown("Run this once from the project folder, then reload the page:")
    st.code("python -m src.ingestion.run_ingest", language="powershell")
    with st.expander("Details"):
        st.caption(str(error))


def render_sources(answer) -> None:
    """One collapsed expander per source, holding the stored question and answer."""
    if not answer.sources:
        return

    st.caption(f"Based on {len(answer.sources)} record(s) from the manual:")
    for chunk in answer.sources:
        label = f"{chunk.record_id} · {chunk.category.replace('_', ' ')} · match {chunk.score:.0%}"
        with st.expander(label):
            st.markdown(
                f'<p class="rg-source-q">{chunk.question}</p>', unsafe_allow_html=True
            )
            st.write(chunk.answer)


def render_refusal() -> None:
    """Neutral, not an error -- refusing correctly is the system working.

    The wording comes from NO_ANSWER_TEXT so the UI and the model's own refusal
    stay the same sentence; the follow-up caption is the UI's to own.
    """
    st.info(NO_ANSWER_TEXT)
    st.caption(
        "Try rephrasing it, use one of the buttons above, or ask a manager if "
        "it's something the manual doesn't cover yet."
    )


def render_answer(answer) -> None:
    if answer.text == EMPTY_QUESTION_TEXT:
        st.caption(EMPTY_QUESTION_TEXT)
        return

    # When a follow-up was rewritten, say so. An employee who sees a surprising
    # answer needs to know what was actually searched for.
    if answer.searched_for:
        st.caption(f"Understood as: {answer.searched_for}")

    if answer.refused:
        render_refusal()
        return

    st.write(answer.text)
    render_sources(answer)


def conversation_pairs() -> list[tuple[str, str]]:
    """Completed exchanges from this session, oldest first.

    The pipeline takes `(question, answer)` pairs; the rendering history is a
    flat list of turns, so pair them up here. Errors are skipped -- a failed
    turn has no answer to resolve a later pronoun against.
    """
    pairs: list[tuple[str, str]] = []
    asked: str | None = None

    for turn in st.session_state.history:
        if turn["role"] == "user":
            asked = turn["content"]
        elif turn["role"] == "assistant" and asked is not None:
            pairs.append((asked, turn["answer"].text))
            asked = None
        elif turn["role"] == "error":
            asked = None

    return pairs


def ask(question: str, retriever: KnowledgeRetriever) -> None:
    """Run one question through the pipeline and record the exchange."""
    history = conversation_pairs()
    st.session_state.history.append({"role": "user", "content": question})

    try:
        with st.spinner("Checking the manual..."):
            answer = answer_question(question, history=history, retriever=retriever)
    except (QueryError, ConfigError) as exc:
        st.session_state.history.append({"role": "error", "content": str(exc)})
        return

    st.session_state.history.append({"role": "assistant", "answer": answer})


def render_history() -> None:
    for turn in st.session_state.history:
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.write(turn["content"])
        elif turn["role"] == "error":
            with st.chat_message("assistant"):
                st.error(turn["content"])
        else:
            with st.chat_message("assistant"):
                render_answer(turn["answer"])


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🍲",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.session_state.setdefault("history", [])

    render_header()

    try:
        retriever = get_retriever()
    except (StoreNotFoundError, ConfigError) as exc:
        # No index means the buttons would only produce errors, so hide them.
        render_sidebar(strategy=None, show_actions=False)
        render_missing_store(exc)
        return

    pending = render_sidebar(strategy=retriever.embedding_strategy)

    typed = st.chat_input("Ask about a procedure, recipe, or customer preference")
    question = typed or pending

    render_history()

    if question:
        ask(question, retriever)
        st.rerun()


if __name__ == "__main__":
    main()
