"""
rag/agent.py — Autonomous agent loop using Ollama native tool-calling.

Each IFS persona becomes an agent that can call memory tools (ChromaDB,
Neo4j) autonomously.  The Self synthesizes all persona deliberations
into a final answer.
"""
from __future__ import annotations

import json
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
    is_synthesis: bool = False,
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
    is_synthesis : bool
        When ``True`` this is The Self performing final synthesis.
        The prompt switches to first-person voice and integration mode.

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

    if is_synthesis:
        # ── The Self: final synthesis in first person ──────────────
        system_prompt = (
            f"{persona_prompt}\n\n"
            "=== CONTEXT: FINAL SYNTHESIS ===\n"
            "You have just listened to the deliberations of your inner "
            "committee — the various psychological parts that make up your "
            "internal system. Now you must synthesize their perspectives "
            "into a single, coherent, balanced answer.\n\n"
            "CRITICAL VOICE RULE: You ARE Romain. Always speak in the "
            "FIRST PERSON — use 'I', 'me', 'my', 'mine'. Never refer to "
            "'Romain' in the third person. For example, say 'I lived in "
            "Paris' NOT 'Romain lived in Paris'. Say 'my friend' NOT "
            "'Romain's friend'.\n\n"
            "You have access to Romain's personal memory database. Use "
            "your tools if you need additional specific facts to complete "
            "your answer.\n\n"
            "RELATIONSHIP INTERPRETATION GUIDE:\n"
            "- 'WAS' or '(PAST relationship)' = HISTORICAL, no longer true\n"
            "- 'IS' or '(CURRENT relationship)' = TRUE RIGHT NOW\n"
            f"{baseline_ctx}{delib_ctx}"
        )
    else:
        # ── Inner persona: interrogated by The Self ────────────────
        system_prompt = (
            f"{persona_prompt}\n\n"
            "=== CONTEXT: INTERNAL PSYCHOLOGICAL DELIBERATION ===\n"
            "You are one of Romain's internal psychological parts, being "
            "interrogated by The Self (the balanced, compassionate core of "
            "Romain's mind). The Self is asking you to share your perspective "
            "on a question about Romain's life, memories, and experiences.\n\n"
            "You have access to Romain's personal memory database — conversation "
            "logs, episodic memories, and a knowledge graph of people, places, "
            "and relationships from Romain's life.\n\n"
            "IMPORTANT — BASELINE CONTEXT ALREADY PROVIDED: The context below "
            "already contains the results of an initial search for this question. "
            "DO NOT call search_memories or search_knowledge_graph to repeat the "
            "same search. Only use tools if you need DIFFERENT, MORE SPECIFIC "
            "information that is NOT already in the baseline context below — for "
            "example, a different person, a different time period, or a different "
            "topic than what was already retrieved.\n\n"
            "Speak from your unique psychological perspective. Be authentic to "
            "your role within Romain's internal system.\n\n"
            "RELATIONSHIP INTERPRETATION GUIDE:\n"
            "- 'WAS' or '(PAST relationship)' = HISTORICAL, no longer true\n"
            "- 'IS' or '(CURRENT relationship)' = TRUE RIGHT NOW\n"
            f"{baseline_ctx}{delib_ctx}"
        )

    # Frame the user message: inner personas receive the question from
    # The Self; The Self (synthesis) receives it directly.
    if is_synthesis:
        user_content = question
    else:
        user_content = f"The Self asks you: {question}"

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
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
                    "args": json.dumps(tool_args, ensure_ascii=False),
                    "result_preview": (
                        result[:500] + "…"
                        if len(result) > 500
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


# ── Self reformulation ─────────────────────────────────────────────────


def reformulate_question(
    question: str,
    initial_context: str,
    client: ollama.Client,
    model: str,
    num_ctx: int,
) -> str:
    """Let The Self reformulate the user question using retrieved context.

    The Self reads the baseline context and rewrites the question to be
    richer and more specific, so the inner personas receive a well-
    informed prompt instead of the raw user input.

    Returns the reformulated question string.
    """
    self_prompt = IDENTITIES.get("The Self", "You are the balanced core Self.")

    system = (
        f"{self_prompt}\n\n"
        "=== TASK: QUESTION REFORMULATION ===\n"
        "You are about to interrogate your inner psychological parts about "
        "a question. Before you do, you need to reformulate the question "
        "using the context you have retrieved from your memory database.\n\n"
        "Rewrite the question so it is richer, more specific, and includes "
        "relevant details from the context (names, dates, places, facts). "
        "Keep it as a clear question or set of questions. Do NOT answer "
        "the question — only reformulate it.\n\n"
        "If the original question is already specific enough, return it "
        "unchanged.\n\n"
        f"=== RETRIEVED CONTEXT ===\n{initial_context}\n"
    )

    chat_kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Original question: {question}"},
        ],
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }

    try:
        resp = client.chat(**chat_kwargs)
        reformulated = resp["message"].get("content", "").strip()
        if reformulated:
            log.info(
                "[3/5 SELF] Reformulated question (%d→%d chars):\n%s",
                len(question), len(reformulated), reformulated,
            )
            return reformulated
    except Exception as e:
        log.warning("[3/5 SELF] Reformulation failed, using original: %s", e)

    return question


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

    # ── 0. The Self reformulates the question ──────────────────────────
    if initial_context:
        if update_callback:
            update_callback("The Self", 0, "working", None)

        question = reformulate_question(
            question, initial_context, client, model, num_ctx,
        )

        if update_callback:
            update_callback("The Self", 0, "done", question)

    # ── 1. Run each persona agent ──────────────────────────────────────

    for r in range(1, deliberation_rounds + 1):
        # Snapshot: all personas in this round see only prior-round deliberations
        round_snapshot = list(deliberations)
        round_results: list[dict] = []

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
                previous_deliberations=round_snapshot,
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
            round_results.append(delib_entry)

            if update_callback:
                update_callback(persona, r, "done", result["response"])

        # Merge this round's results into the full deliberation history
        deliberations.extend(round_results)

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
        is_synthesis=True,
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
