"""
rag/llm.py — Ollama LLM call with thinking-token support.
"""
import json
import logging

import ollama
from config import IDENTITIES

log = logging.getLogger(__name__)

def _build_context_string(docs: list, episodes: list, facts: list) -> str:
    ctx = ""
    if facts:
        ctx += "=== SEMANTIC FACTS (Neo4j) ===\n"
        ctx += "\n".join(f"- {f}" for f in facts)
        ctx += "\n\n"
    if episodes:
        ctx += "=== EPISODIC MEMORY (Chroma) ===\n"
        ctx += "\n".join(f"[{e['date']}] {e['content']}" for e in episodes)
        ctx += "\n\n"
    if docs:
        ctx += "=== CONVERSATION LOGS (Chroma) ===\n"
        ctx += "\n\n---\n\n".join(
            f"[{d['date']}] [Chat: {d['friend']}]\n{d['content']}" for d in docs
        )
    return ctx

def _safe_chat(client, kwargs):
    """
    Executes client.chat(**kwargs). If the model rejects the 'think' parameter,
    it dynamically strips it and retries.
    """
    try:
        return client.chat(**kwargs)
    except ollama.ResponseError as e:
        if e.status_code == 400 and "does not support thinking" in str(e).lower():
            if "think" in kwargs:
                kwargs.pop("think")
            return client.chat(**kwargs)
        raise

def _parse_llm_response(resp: dict):
    msg     = resp["message"]
    content = msg.get("content", "")
    thinking = ""
    answer   = content

    # Method 1: Ollama ≥ 0.6 returns thinking in a dedicated field
    dedicated = msg.get("thinking", "")
    if dedicated and dedicated.strip():
        thinking = dedicated.strip()
        answer   = content.strip()

    # Method 2 (fallback): thinking embedded as <think>…</think> in content
    elif "<think>" in content and "</think>" in content:
        ts = content.find("<think>") + 7
        te = content.find("</think>")
        thinking = content[ts:te].strip()
        answer   = content[te + 8:].strip()

    prompt_tok = resp.get("prompt_eval_count", 0)
    comp_tok   = resp.get("eval_count", 0)
    return thinking, answer, prompt_tok, comp_tok

def filter_irrelevant_context(question: str, docs: list, model: str, ollama_host: str) -> list:
    """
    Passes each retrieved conversation chunk through a lightweight LLM to determine
    if it actually contains meaningful context to answer the user's question, or if
    it was just a coincidental keyword match.
    """
    if not docs:
        return docs
        
    client = ollama.Client(host=ollama_host)
    filtered_docs = []
    
    log.debug("Context purification (%s): %d docs to filter", model, len(docs))
    
    for i, doc in enumerate(docs):
        prompt = f"""
You are a context purificaton filter for a RAG system.
Your job is to read a chunk of an old chat log and determine if it contains ANY relevant information that could help answer the user's question.

User Question: "{question}"

Chat Log Snippet:
{doc['content']}

Does this chat snippet contain anything tangibly relevant to the user's question?
Return ONLY valid JSON matching this schema:
{{
  "is_relevant": true or false
}}
""".strip()

        try:
            log.debug("Doc %d purification prompt:\n%s", i + 1, prompt)
            res = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0}
            )
            content = res["message"]["content"]
            result = json.loads(content)
            
            is_relevant = result.get("is_relevant", True)
            log.debug("Doc %d relevant: %s", i + 1, is_relevant)
            
            if is_relevant:
                filtered_docs.append(doc)
        except Exception as e:
            log.warning("Context purification failed for doc %d, keeping: %s", i + 1, e)
            filtered_docs.append(doc)

    log.info(
        "[3/5 PURIFY] kept %d / %d docs (removed %d irrelevant)",
        len(filtered_docs), len(docs), len(docs) - len(filtered_docs),
    )
    return filtered_docs

def call_llm(question: str, docs: list, episodes: list, facts: list,
             model: str, ollama_host: str, num_ctx: int, system_prompt: str, enable_thinking: bool = True):
    """
    Returns (thinking, answer, prompt_tokens, completion_tokens).
    """
    ctx = _build_context_string(docs, episodes, facts)
    full_system_prompt = (
        f"{system_prompt}\n\n"
        f"CRITICAL VOICE RULE: You ARE Romain. Always speak in the "
        f"FIRST PERSON — use 'I', 'me', 'my', 'mine'. Never refer to "
        f"'Romain' in the third person. For example, say 'I lived in "
        f"Paris' NOT 'Romain lived in Paris'. Say 'my friend' NOT "
        f"'Romain's friend'.\n\n"
        f"=== RELATIONSHIP INTERPRETATION GUIDE ===\n"
        f"Pay extremely close attention to the tense of semantic facts from the graph.\n"
        f"- 'WAS' or '(PAST relationship)' means the state is HISTORICAL and NO LONGER TRUE.\n"
        f"- 'IS' or '(CURRENT relationship)' means the state is TRUE RIGHT NOW.\n"
        f"Do not confuse past relationships with current ones.\n\n"
        f"CONTEXT:\n{ctx}"
    )

    client = ollama.Client(host=ollama_host)
    chat_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": full_system_prompt},
            {"role": "user",   "content": question},
        ],
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }
    if enable_thinking:
        chat_kwargs["think"] = True

    resp = _safe_chat(client, chat_kwargs)

    return _parse_llm_response(resp)

def deliberate_and_synthesize(question: str, docs: list, episodes: list, facts: list,
                              model: str, ollama_host: str, num_ctx: int,
                              active_personas: list, deliberation_rounds: int, enable_thinking: bool = True,
                              update_callback=None, tool_executor=None, cancel_check=None):
    """
    Returns (thinking, answer, prompt_tokens, completion_tokens, deliberations).

    When *tool_executor* is provided the new agent-loop path is used:
    each persona becomes an autonomous tool-calling agent that decides
    for itself whether to search memories / the knowledge graph.

    When *tool_executor* is ``None`` the legacy static-context path is
    used as a fallback (identical to the pre-agent-loop behaviour).
    """
    # ── New path: agent loop ───────────────────────────────────────────
    if tool_executor is not None:
        from rag.agent import run_committee

        # Build initial context from upfront naive retrieval
        initial_context = _build_context_string(docs, episodes, facts)

        return run_committee(
            question=question,
            active_personas=active_personas,
            model=model,
            ollama_host=ollama_host,
            num_ctx=num_ctx,
            tool_executor=tool_executor,
            deliberation_rounds=deliberation_rounds,
            enable_thinking=enable_thinking,
            update_callback=update_callback,
            initial_context=initial_context,
            cancel_check=cancel_check,
        )

    # ── Legacy path: static context (fallback) ─────────────────────────
    ctx = _build_context_string(docs, episodes, facts)
    client = ollama.Client(host=ollama_host)

    total_prompt_tok = 0
    total_comp_tok = 0
    deliberations = []

    # 1. Deliberation Rounds
    for r in range(1, deliberation_rounds + 1):
        # Snapshot: all personas in this round see only prior-round deliberations
        round_snapshot = list(deliberations)
        round_results: list[dict] = []

        for persona in active_personas:
            if persona not in IDENTITIES:
                continue

            persona_sys_prompt = IDENTITIES[persona]

            # Inject previous deliberations into the context for this persona
            delib_ctx = ""
            if round_snapshot:
                delib_ctx = "\n\n=== PREVIOUS DELIBERATIONS FROM OTHER PERSONAS ===\n"
                for d in round_snapshot:
                    delib_ctx += f"[{d['persona']} - Round {d['round']}]: {d['response']}\n\n"

            full_system_prompt = (
                f"{persona_sys_prompt}\n\n"
                f"=== CONTEXT: INTERNAL PSYCHOLOGICAL DELIBERATION ===\n"
                f"You are one of Romain's internal psychological parts, being "
                f"interrogated by The Self (the balanced, compassionate core of "
                f"Romain's mind). The Self is asking you to share your perspective "
                f"on a question about Romain's life, memories, and experiences.\n\n"
                f"You have access to Romain's personal memory database — conversation "
                f"logs, episodic memories, and a knowledge graph of people, places, "
                f"and relationships from Romain's life.\n\n"
                f"Speak from your unique psychological perspective. Be authentic to "
                f"your role within Romain's internal system. Keep your answer focused.\n\n"
                f"=== RELATIONSHIP INTERPRETATION GUIDE ===\n"
                f"Pay extremely close attention to the tense of semantic facts from the graph.\n"
                f"- 'WAS' or '(PAST relationship)' means the state is HISTORICAL and NO LONGER TRUE.\n"
                f"- 'IS' or '(CURRENT relationship)' means the state is TRUE RIGHT NOW.\n"
                f"Do not confuse past relationships with current ones.\n\n"
                f"CONTEXT:\n{ctx}{delib_ctx}"
            )

            chat_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user",   "content": f"The Self asks you: {question}"},
                ],
                "stream": False,
                "options": {"num_ctx": num_ctx},
            }

            log.debug("Legacy deliberation: %s round %d", persona, r)

            if enable_thinking:
                chat_kwargs["think"] = True

            if update_callback:
                update_callback(persona, r, "working", None)

            resp = _safe_chat(client, chat_kwargs)

            thinking, answer, p_tok, c_tok = _parse_llm_response(resp)
            total_prompt_tok += p_tok
            total_comp_tok += c_tok

            if update_callback:
                update_callback(persona, r, "done", answer)

            round_results.append({
                "persona": persona,
                "round": r,
                "response": answer,
                "tool_trace": [],
            })

        # Merge this round's results into the full deliberation history
        deliberations.extend(round_results)

    # 2. Final Synthesis by "The Self"
    synthesis_sys_prompt = IDENTITIES.get("The Self", "You are the balanced core Self.")
    delib_ctx = "\n\n=== INNER COMMITTEE DELIBERATION ===\n"
    if deliberations:
        for d in deliberations:
            delib_ctx += f"[{d['persona']} - Round {d['round']}]: {d['response']}\n\n"
    else:
        delib_ctx += "No other personas participated.\n\n"

    full_system_prompt = (
        f"{synthesis_sys_prompt}\n\n"
        f"=== CONTEXT: FINAL SYNTHESIS ===\n"
        f"You have listened to the deliberations of your inner committee. "
        f"Now, synthesize a final, balanced, and coherent answer to the user's question, "
        f"taking into account the various perspectives but speaking with one unified voice.\n\n"
        f"CRITICAL VOICE RULE: You ARE Romain. Always speak in the "
        f"FIRST PERSON — use 'I', 'me', 'my', 'mine'. Never refer to "
        f"'Romain' in the third person. For example, say 'I lived in "
        f"Paris' NOT 'Romain lived in Paris'. Say 'my friend' NOT "
        f"'Romain's friend'.\n\n"
        f"=== RELATIONSHIP INTERPRETATION GUIDE ===\n"
        f"Pay extremely close attention to the tense of semantic facts from the graph.\n"
        f"- 'WAS' or '(PAST relationship)' means the state is HISTORICAL and NO LONGER TRUE.\n"
        f"- 'IS' or '(CURRENT relationship)' means the state is TRUE RIGHT NOW.\n"
        f"Do not confuse past relationships with current ones.\n\n"
        f"CONTEXT:\n{ctx}{delib_ctx}"
    )

    chat_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": full_system_prompt},
            {"role": "user",   "content": question},
        ],
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }

    log.debug("Legacy synthesis: The Self")

    if enable_thinking:
        chat_kwargs["think"] = True

    resp = _safe_chat(client, chat_kwargs)

    thinking, answer, p_tok, c_tok = _parse_llm_response(resp)
    total_prompt_tok += p_tok
    total_comp_tok += c_tok

    return thinking, answer, total_prompt_tok, total_comp_tok, deliberations
