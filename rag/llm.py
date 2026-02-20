"""
rag/llm.py — Ollama LLM call with thinking-token support.
"""
import ollama


def call_llm(question: str, docs: list, episodes: list,
             model: str, ollama_host: str, num_ctx: int):
    """
    Returns (thinking, answer, prompt_tokens, completion_tokens).

    Builds a context string from episodic memory + chat chunks, then calls
    the Ollama API. Supports both dedicated thinking field (Ollama ≥ 0.6)
    and legacy <think>…</think> tag parsing.
    """
    # Build context string
    ctx = ""
    if episodes:
        ctx += "=== EPISODIC MEMORY ===\n"
        ctx += "\n".join(f"[{e['date']}] {e['content']}" for e in episodes)
        ctx += "\n\n"
    ctx += "=== CHAT LOGS ===\n"
    ctx += "\n\n---\n\n".join(
        f"[{d['date']}] [Chat: {d['friend']}]\n{d['content']}" for d in docs
    )

    system_prompt = (
        "You are an AI assistant analysing personal Facebook history. "
        "You have access to both high-level life episodes and detailed chat logs. "
        "Answer in the same language as the user's question. "
        "Be direct and specific.\n\n"
        f"CONTEXT:\n{ctx}"
    )

    client = ollama.Client(host=ollama_host)
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        stream=False,
        options={"num_ctx": num_ctx},
        think=True,   # request thinking tokens (Ollama ≥ 0.6, ignored by older versions)
    )

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
