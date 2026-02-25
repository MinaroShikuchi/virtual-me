"""
rag/llm.py — Ollama LLM call with thinking-token support.
"""
import ollama
from config import IDENTITIES

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

import json

def translate_to_introspective(question: str, model: str, ollama_host: str) -> str:
    """
    Translates a 2nd-person query into a 1st-person introspective query.
    E.g. "What do you know about Lois normand?" -> "What do I know about Lois normand?"
    Returns the mapped string.
    """
    prompt = f"""
You are a perspective translator. Rewrite the user's question from a 2nd-person external query into a 1st-person introspective internal thought.
Change "you" to "I", "your" to "my", etc.
Do not answer the question. ONLY return the translated text.

Example:
Input: What do you know about Lois normand?
Output: What do I know about Lois normand?

Input: Where did you travel last year?
Output: Where did I travel last year?

Input: {question}
Output:
""".strip()

    client = ollama.Client(host=ollama_host)
    try:
        res = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        return res["message"]["content"].strip()
    except Exception as e:
        print(f"Failed to translate query: {e}")
        return question

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
    
    print(f"\n{'='*50}\n[DEBUG: CONTEXT PURIFICATION ({model})]\n{'='*50}")
    
    for i, doc in enumerate(docs):
        meta_str = f"[{doc.get('date', 'N/A')}] {doc.get('friend', 'N/A')}"
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
            print(f"\n[Doc {i+1} ({meta_str}) PROMPT]:\n{prompt}\n")
            res = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0}
            )
            content = res["message"]["content"]
            result = json.loads(content)
            
            print(f"[Doc {i+1} OUTPUT]:\n{json.dumps(result, indent=2)}\n")
            
            is_relevant = result.get("is_relevant", True)
            print(f"Doc {i+1} ({meta_str}) relevant: {is_relevant}")
            
            if is_relevant:
                filtered_docs.append(doc)
        except Exception as e:
            print(f"Context purification failed for doc {i+1}, keeping it by default: {e}")
            filtered_docs.append(doc)
            
    print(f"Purification completed: kept {len(filtered_docs)} out of {len(docs)} docs\n{'='*50}\n")
    return filtered_docs

def call_llm(question: str, docs: list, episodes: list, facts: list,
             model: str, ollama_host: str, num_ctx: int, system_prompt: str, enable_thinking: bool = True):
    """
    Returns (thinking, answer, prompt_tokens, completion_tokens).
    """
    ctx = _build_context_string(docs, episodes, facts)
    full_system_prompt = (
        f"{system_prompt}\n\n"
        f"CRITICAL PERSPECTIVE RULE:\n"
        f"You are the core of our mind. When the context mentions 'Romain', it is talking about US/ME. We are Romain. Always refer to 'Romain' as 'me', 'us', 'we', or 'I'. Never refer to Romain in the third person.\n\n"
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
                              update_callback=None):
    """
    Returns (thinking, answer, prompt_tokens, completion_tokens, deliberations).
    Deliberations is a list of dicts: {"persona": name, "round": r, "response": text}
    """
    ctx = _build_context_string(docs, episodes, facts)
    client = ollama.Client(host=ollama_host)
    
    print(f"\n[DELIBERATE] Starting committee with {len(docs)} docs, {len(episodes)} episodes, {len(facts)} facts.")
    
    total_prompt_tok = 0
    total_comp_tok = 0
    deliberations = []
    
    # 1. Deliberation Rounds
    for r in range(1, deliberation_rounds + 1):
        for persona in active_personas:
            if persona not in IDENTITIES:
                continue
                
            persona_sys_prompt = IDENTITIES[persona]
            
            # Inject previous deliberations into the context for this persona
            delib_ctx = ""
            if deliberations:
                delib_ctx = "\n\n=== PREVIOUS DELIBERATIONS FROM OTHER PERSONAS ===\n"
                for d in deliberations:
                    delib_ctx += f"[{d['persona']} - Round {d['round']}]: {d['response']}\n\n"
                    
            full_system_prompt = (
                f"{persona_sys_prompt}\n\n"
                f"You are participating in an inner committee to help answer a question. "
                f"Provide your perspective based on your psychological identity. Keep your answer focused.\n\n"
                f"CRITICAL PERSPECTIVE RULE:\n"
                f"You are inside our mind. When the retrieved context (conversations, facts, episodes) mentions 'Romain', it is talking about US. We are Romain. Always refer to 'Romain' as 'us', 'we', or 'I' (from the collective perspective of the system). Never refer to Romain in the third person.\n\n"
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
            
            print(f"\n{'='*50}\n[DEBUG: INPUT TO {persona.upper()} (ROUND {r})]\n{'='*50}")
            # print(f"SYSTEM PROMPT:\n{full_system_prompt}") # Redundant, context already logged
            print(f"IDENTITY PROMPT:\n{persona_sys_prompt}")
            # print(f"\nUSER QUESTION:\n{question}\n{'='*50}\n")  # Redundant, logged already
            
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
                
            deliberations.append({
                "persona": persona,
                "round": r,
                "response": answer
            })
            
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
        f"You have listened to the deliberations of your inner committee. "
        f"Now, synthesize a final, balanced, and coherent answer to the user's question, "
        f"taking into account the various perspectives but speaking with one unified voice.\n\n"
        f"CRITICAL PERSPECTIVE RULE:\n"
        f"You are the core of our mind. When the context or deliberations mention 'Romain', it is talking about US/ME. We are Romain. Always refer to 'Romain' as 'me', 'us', 'we', or 'I'. Never refer to Romain in the third person.\n\n"
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
    
    print(f"\n{'='*50}\n[DEBUG: INPUT TO THE SELF (FINAL SYNTHESIS)]\n{'='*50}")
    # print(f"SYSTEM PROMPT:\n{full_system_prompt}") # Redundant
    print(f"IDENTITY PROMPT:\n{synthesis_sys_prompt}")
    # print(f"\nUSER QUESTION:\n{question}\n{'='*50}\n") # Redundant
    
    if enable_thinking:
        chat_kwargs["think"] = True
        
    resp = _safe_chat(client, chat_kwargs)
    
    thinking, answer, p_tok, c_tok = _parse_llm_response(resp)
    total_prompt_tok += p_tok
    total_comp_tok += c_tok
    
    return thinking, answer, total_prompt_tok, total_comp_tok, deliberations
