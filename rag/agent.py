"""
rag/agent.py — Autonomous agent loop using Ollama native tool-calling.

Each IFS persona becomes an agent that can call memory tools (ChromaDB,
Neo4j) autonomously.  The Self synthesizes all persona deliberations
into a final answer.
"""
from __future__ import annotations

import logging

import ollama
from config import IDENTITIES
from rag.tools import TOOL_SCHEMAS, ToolExecutor

log = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 3  # Per-persona cap on tool-calling rounds


# ── Single persona agent ───────────────────────────────────────────────


def _run_persona_agent(
    persona: str,
    question: str,
    client: ollama.Client,
    model: str,
    num_ctx: int,
    tool_executor: ToolExecutor,
    previous_deliberations: list[dict],
    enable_thinking: bool = True,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    initial_context: str = "",
) -> dict:
    """Run a single persona as a tool-calling agent.

    The agent receives the user question plus tool definitions.  It may
    call tools zero or more times (up to *max_iterations*) before
    producing its final textual response.

    Parameters
    ----------
    initial_context : str
        Pre-retrieved context from upfront naive retrieval.  Injected
        into the system prompt so the persona has baseline knowledge
        before deciding whether to call additional tools.

    Returns
    -------
    dict
        Keys: persona, response, thinking, tool_trace,
              prompt_tokens, completion_tokens
    """
    persona_prompt = IDENTITIES[persona]

    # Build context from earlier deliberations (if any)
    delib_ctx = ""
    if previous_deliberations:
        delib_ctx = "\n\n=== PREVIOUS DELIBERATIONS ===\n"
        for d in previous_deliberations:
            delib_ctx += f"[{d['persona']}]: {d['response']}\n\n"

    # Build baseline context block from upfront retrieval
    baseline_ctx = ""
    if initial_context:
        baseline_ctx = (
            "\n\n=== BASELINE CONTEXT (from initial retrieval) ===\n"
            f"{initial_context}\n"
        )

    system_prompt = (
        f"{persona_prompt}\n\n"
        "You are participating in an inner committee deliberation. "
        "You have access to memory tools — use them if you need to recall "
        "specific facts, conversations, or experiences to inform your perspective. "
        "Some baseline context has already been retrieved for you below. "
        "Use your tools only if you need MORE specific or targeted information "
        "beyond what is already provided.\n\n"
        "RELATIONSHIP INTERPRETATION GUIDE:\n"
        "- 'WAS' or '(PAST relationship)' = HISTORICAL, no longer true\n"
        "- 'IS' or '(CURRENT relationship)' = TRUE RIGHT NOW\n"
        f"{baseline_ctx}{delib_ctx}"
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Log what's being passed to this agent
    tool_names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    log.info(
        "  ├─ system prompt: %d chars  |  baseline context: %d chars  |  "
        "prev deliberations: %d  |  tools: %s",
        len(system_prompt),
        len(initial_context),
        len(previous_deliberations),
        tool_names,
    )

    tool_trace: list[dict] = []
    total_p_tok = 0
    total_c_tok = 0
    msg: dict = {}

    for _iteration in range(max_iterations):
        chat_kwargs: dict = {
            "model": model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "stream": False,
            "options": {"num_ctx": num_ctx},
        }
        if enable_thinking:
            chat_kwargs["think"] = True

        try:
            resp = client.chat(**chat_kwargs)
        except ollama.ResponseError as e:
            # Model doesn't support tool-calling or thinking — strip and retry
            if e.status_code == 400:
                err_msg = str(e).lower()
                if "does not support tools" in err_msg:
                    chat_kwargs.pop("tools", None)
                if "does not support thinking" in err_msg:
                    chat_kwargs.pop("think", None)
                resp = client.chat(**chat_kwargs)
            else:
                raise

        msg = resp["message"]
        total_p_tok += resp.get("prompt_eval_count", 0)
        total_c_tok += resp.get("eval_count", 0)

        # Check for tool calls
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break  # Model is done — has its final answer

        # Append the assistant message (contains tool_calls) to history
        messages.append(msg)

        # Execute each tool call and feed results back
        for tc in tool_calls:
            fn = tc["function"]
            tool_name = fn["name"]
            tool_args = fn.get("arguments", {})

            log.info("  [TOOL] %s calling %s(%s)", persona, tool_name, tool_args)

            result = tool_executor.execute(tool_name, tool_args)

            tool_trace.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "result_preview": (
                        result[:200] + "…"
                        if len(result) > 200
                        else result
                    ),
                }
            )

            messages.append({"role": "tool", "content": result})

    # Parse final response
    content = msg.get("content", "") if msg else ""
    thinking = msg.get("thinking", "") if msg else ""

    # Handle <think> tags embedded in content (older Ollama versions)
    if not thinking and "<think>" in content and "</think>" in content:
        ts = content.find("<think>") + 7
        te = content.find("</think>")
        thinking = content[ts:te].strip()
        content = content[te + 8:].strip()

    return {
        "persona": persona,
        "response": content,
        "thinking": thinking,
        "tool_trace": tool_trace,
        "prompt_tokens": total_p_tok,
        "completion_tokens": total_c_tok,
    }


# ── Committee orchestrator ─────────────────────────────────────────────


class CancelledError(Exception):
    """Raised when the user cancels generation."""


def run_committee(
    question: str,
    active_personas: list[str],
    model: str,
    ollama_host: str,
    num_ctx: int,
    tool_executor: ToolExecutor,
    deliberation_rounds: int = 1,
    enable_thinking: bool = True,
    update_callback=None,
    initial_context: str = "",
    cancel_check=None,
) -> tuple:
    """Orchestrate the full committee deliberation.

    Each active persona runs as a tool-calling agent, then The Self
    synthesizes all deliberations into a final answer.

    Parameters
    ----------
    question : str
        The user's question.
    active_personas : list[str]
        Persona names to participate (must exist in ``IDENTITIES``).
    model, ollama_host, num_ctx : str, str, int
        Ollama connection / model settings.
    tool_executor : ToolExecutor
        Pre-configured executor for memory tools.
    deliberation_rounds : int
        How many rounds of deliberation to run.
    enable_thinking : bool
        Whether to request thinking tokens from the model.
    update_callback : callable | None
        ``(persona, round, status, response)`` callback for UI updates.
    initial_context : str
        Pre-retrieved context from upfront naive retrieval, passed
        through to each persona agent.
    cancel_check : callable | None
        Zero-arg callable returning ``True`` when the user has requested
        cancellation.  Checked before each persona run.

    Returns
    -------
    tuple
        ``(thinking, answer, prompt_tokens, completion_tokens, deliberations)``
    """
    client = ollama.Client(host=ollama_host)
    deliberations: list[dict] = []
    total_p_tok = 0
    total_c_tok = 0

    # ── 1. Run each persona agent ──────────────────────────────────────

    for r in range(1, deliberation_rounds + 1):
        for persona in active_personas:
            if cancel_check and cancel_check():
                raise CancelledError("Generation cancelled by user.")

            if persona not in IDENTITIES:
                continue

            if update_callback:
                update_callback(persona, r, "working", None)

            log.info("[4/5 AGENT] %s — Round %d", persona.upper(), r)

            result = _run_persona_agent(
                persona=persona,
                question=question,
                client=client,
                model=model,
                num_ctx=num_ctx,
                tool_executor=tool_executor,
                previous_deliberations=deliberations,
                enable_thinking=enable_thinking,
                initial_context=initial_context,
            )

            total_p_tok += result["prompt_tokens"]
            total_c_tok += result["completion_tokens"]

            delib_entry = {
                "persona": persona,
                "round": r,
                "response": result["response"],
                "tool_trace": result["tool_trace"],
            }
            deliberations.append(delib_entry)

            if update_callback:
                update_callback(persona, r, "done", result["response"])

    # ── 2. The Self synthesizes ────────────────────────────────────────

    if cancel_check and cancel_check():
        raise CancelledError("Generation cancelled by user.")

    if update_callback:
        update_callback("The Self", deliberation_rounds + 1, "working", None)

    log.info("[5/5 SYNTHESIS] THE SELF — Final Synthesis")

    self_result = _run_persona_agent(
        persona="The Self",
        question=question,
        client=client,
        model=model,
        num_ctx=num_ctx,
        tool_executor=tool_executor,
        previous_deliberations=deliberations,
        enable_thinking=enable_thinking,
        initial_context=initial_context,
    )

    total_p_tok += self_result["prompt_tokens"]
    total_c_tok += self_result["completion_tokens"]

    if update_callback:
        update_callback(
            "The Self",
            deliberation_rounds + 1,
            "done",
            self_result["response"],
        )

    return (
        self_result["thinking"],
        self_result["response"],
        total_p_tok,
        total_c_tok,
        deliberations,
    )
