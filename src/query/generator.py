"""Answer generation with gemini-3.5-flash-lite at temperature=0.

Determinism is load-bearing: the accuracy benchmark is only meaningful if the
same question yields the same answer on every run.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from src import config
from src.query.prompts import SYSTEM_PROMPT
from src.query.retriever import QueryError

RETRY_DELAY_SECONDS = 2


def get_llm() -> ChatGoogleGenerativeAI:
    """Build the generation client."""
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.require_api_key(),
    )


def generate(
    prompt: str,
    llm: ChatGoogleGenerativeAI | None = None,
    *,
    system: str = SYSTEM_PROMPT,
) -> str:
    """Run the guarded prompt and return the answer text.

    On failure the caller gets an error, never a fallback answer: answering
    from the model's own knowledge is precisely the behaviour this system
    exists to prevent.

    `system` defaults to the answering guardrail. The condenser passes its own
    system prompt so it can share this retry and empty-response handling
    without inheriting rules written for answering.
    """
    import time

    client = llm or get_llm()
    messages = [("system", system), ("human", prompt)]
    last_error: Exception | None = None

    for attempt in (1, 2):
        try:
            response = client.invoke(messages)
            text = (response.content or "").strip()
            if not text:
                raise QueryError("the assistant returned an empty response")
            return text
        except QueryError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider raises a wide range
            last_error = exc
            if attempt == 1:
                time.sleep(RETRY_DELAY_SECONDS)

    raise QueryError(
        f"the assistant is unreachable right now -- check your connection and "
        f"try again ({last_error})"
    ) from last_error
