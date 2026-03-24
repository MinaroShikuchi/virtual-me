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

def condense_context(question: str, raw_context: str, model: str, ollama_host: str) -> tuple[str, str]:
    """
    Uses a fast intent model to summarize the context, extracting only the most relevant points.
    Returns (condensed_text, stats_string).
    """
    if not raw_context.strip():
        return "", "Context was empty."
        
    print(f"  -> Context Condenser: Summarizing context using {model}...")
    prompt = f"""
You are a context condenser for a memory-augmented AI. 
The retrieved context below is too long for the current processing window.
Your task is to summarize the following context into its most relevant points to answer the user's question: "{question}"

Rules:
1. Maintain all specific dates, names, and key quantitative facts (e.g., song counts, game sessions).
2. Group related information logically.
3. Be as concise as possible without losing the 'ground truth' of the data.
4. If there are conflicting facts (e.g. past vs current relationships), preserve both but label them clearly.
5. DO NOT return an empty string if the context has content. If unsure, just repeat the most important parts.

RAW CONTEXT:
{raw_context}

CONDENSED MEMORY SUMMARY:
""".strip()

    client = ollama.Client(host=ollama_host)
    try:
        res = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        result = res["message"]["content"].strip()
        if not result:
             return raw_context, "Condenser model returned empty result - using raw."
        
        stats = f"Condensed {len(raw_context)} -> {len(result)} chars ({(len(result)/len(raw_context))*100:.1f}%)"
        return result, stats
    except Exception as e:
        print(f"Condenser failed: {e}")
        return raw_context, f"Condenser failed: {e}"

def _safe_chat(client, kwargs):
    """
    Executes client.chat(**kwargs). If the model rejects the 'think' parameter
    or 'tools' parameter, it dynamically strips them and retries.
    """
    try:
        return client.chat(**kwargs)
    except ollama.ResponseError as e:
        err_str = str(e).lower()
        if e.status_code == 400 and "does not support thinking" in err_str:
            kwargs.pop("think", None)
            return client.chat(**kwargs)
        if e.status_code == 400 and ("tool" in err_str or "function" in err_str):
            kwargs.pop("tools", None)
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
If the input is a simple greeting or doesn't have 2nd-person pronouns, leave it alone.
Do not answer the question. ONLY return the translated text.

Example:
Input: What do you know about photography?
Output: What do I know about photography?

Input: Where did you travel last year?
Output: Where did I travel last year?

Input: Hello there!
Output: Hello there!

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

def _build_history_messages(conversation_history: list | None) -> list:
    """Extract prior user/assistant turns for multi-turn context."""
    if not conversation_history:
        return []
    msgs = []
    for msg in conversation_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    return msgs


def call_llm(question: str, docs: list, episodes: list, facts: list,
             model: str, ollama_host: str, num_ctx: int, system_prompt: str,
             enable_thinking: bool = True, num_predict: int = 1024,
             conversation_history: list | None = None,
             tool_callback=None,
             enable_condenser: bool = False,
             condenser_threshold: int = 70,
             intent_model: str = "llama3.2"):
    """
    Returns (thinking, answer, prompt_tokens, completion_tokens, duration, context_used, condenser_stats).
    Supports Ollama native tool calling — if the model invokes a skill, it runs
    the function, injects the result, and calls the model again for a final answer.
    """
    from rag.skills import OLLAMA_TOOLS, SKILLS_REGISTRY
    import json

    ctx = _build_context_string(docs, episodes, facts)
    history_text = "\n".join([m.get("content", "") for m in (conversation_history or [])])
    total_est = (len(ctx) + len(history_text) + len(system_prompt) + (num_predict * 4)) // 3
    
    # Auto Context Condenser
    condenser_stats = "Not triggered"
    if enable_condenser and total_est > (num_ctx * condenser_threshold / 100):
        ctx, condenser_stats = condense_context(question, ctx, intent_model, ollama_host)
        print(f"[DEBUG] Condenser Stats: {condenser_stats}")
    full_system_prompt = (
        f"{system_prompt}\n\n"
        f"CRITICAL PERSPECTIVE RULE:\n"
        f"You are the core of my mind. When the context mentions 'Romain', it is talking about ME. I am Romain. Always refer to 'Romain' as 'me' or 'I'. Never refer to 'Romain' in the third person.\n\n"
        f"=== RELATIONSHIP INTERPRETATION GUIDE ===\n"
        f"Pay extremely close attention to the tense of semantic facts from the graph.\n"
        f"- 'WAS' or '(PAST relationship)' means the state is HISTORICAL and NO LONGER TRUE.\n"
        f"- 'IS' or '(CURRENT relationship)' means the state is TRUE RIGHT NOW.\n"
        f"Do not confuse past relationships with current ones.\n\n"
        f"CONTEXT:\n{ctx}"
    )

    messages = [{"role": "system", "content": full_system_prompt}]
    messages.extend(_build_history_messages(conversation_history))
    messages.append({"role": "user", "content": question})

    client = ollama.Client(host=ollama_host)
    
    # Tools + thinking mode often conflict — prefer tools when thinking is OFF
    # When thinking is ON, tools are disabled but we rely on intent-based skill execution
    use_tools = not enable_thinking
    
    chat_kwargs = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict
        },
    }
    if use_tools:
        chat_kwargs["tools"] = OLLAMA_TOOLS
    if enable_thinking:
        chat_kwargs["think"] = True

    import time
    start_t = time.perf_counter()
    resp = _safe_chat(client, chat_kwargs)

    # ── Tool Call Handling ────────────────────────────────────────────────────
    tool_calls = resp.get("message", {}).get("tool_calls", [])
    if tool_calls:
        print(f"\n[TOOL CALLING] Model requested {len(tool_calls)} tool(s):")
        # Append the model's tool-calling message
        messages.append(resp["message"])

        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", {})
            
            # Normalize arguments
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}
            else:
                args = raw_args
            
            print(f"  -> Tool: {name}({args})")
            if tool_callback:
                tool_callback(name, args)

            if name in SKILLS_REGISTRY:
                try:
                    result = SKILLS_REGISTRY[name](**args)
                    result_str = "\n".join(result) if result else "No results found."
                    print(f"  -> Tool Result ({name}):\n{result_str}")
                except Exception as e:
                    result_str = f"Error executing {name}: {e}"
                    print(f"  -> Tool Error: {e}")
            else:
                result_str = f"Unknown tool: {name}"

            # Inject tool result as a tool role message
            messages.append({
                "role": "tool",
                "content": result_str,
            })

        # Second LLM call with tool results injected
        chat_kwargs_final = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict
            },
        }
        if enable_thinking:
            chat_kwargs_final["think"] = True

        resp = _safe_chat(client, chat_kwargs_final)

    duration = time.perf_counter() - start_t

    thinking, answer, p_tok, c_tok = _parse_llm_response(resp)
    return thinking, answer, p_tok, c_tok, duration, ctx, condenser_stats

def deliberate_and_synthesize(question: str, docs: list, episodes: list, facts: list,
                               model: str, ollama_host: str, num_ctx: int,
                               active_personas: list, deliberation_rounds: int, 
                               enable_thinking: bool = True, num_predict: int = 1024,
                               update_callback=None, conversation_history: list | None = None,
                               discovered_identities: dict | None = None,
                               enable_condenser: bool = False,
                               condenser_threshold: int = 70,
                               intent_model: str = "llama3.2"):
    """
    Returns (thinking, answer, prompt_tokens, completion_tokens, deliberations).
    Deliberations is a list of dicts: {"persona": name, "round": r, "response": text}

    Parameters
    ----------
    discovered_identities : dict | None
        Optional dict mapping identity name → description string.
        These are merged with the hardcoded IDENTITIES so that
        discovered personality facets can participate in deliberation.
    """
    ctx = _build_context_string(docs, episodes, facts)
    history_text = "\n".join([m.get("content", "") for m in (conversation_history or [])])
    total_est = (len(ctx) + len(history_text) + 1000 + (num_predict * 4)) // 3
    
    # Auto Context Condenser
    condenser_stats = "Not triggered"
    if enable_condenser and total_est > (num_ctx * condenser_threshold / 100):
        ctx, condenser_stats = condense_context(question, ctx, intent_model, ollama_host)
        print(f"[DEBUG] Condenser Stats: {condenser_stats}")

    client = ollama.Client(host=ollama_host)

    import time
    start_t = time.perf_counter()

    # Merge hardcoded identities with discovered ones
    all_identities = dict(IDENTITIES)
    if discovered_identities:
        all_identities.update(discovered_identities)
    
    print(f"\n[DELIBERATE] Starting committee with {len(docs)} docs, {len(episodes)} episodes, {len(facts)} facts.")
    
    total_prompt_tok = 0
    total_comp_tok = 0
    deliberations = []
    
    # 1. Deliberation Rounds
    for r in range(1, deliberation_rounds + 1):
        for persona in active_personas:
            if persona not in all_identities:
                continue
                
            persona_sys_prompt = all_identities[persona]
            
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
                f"You are inside our mind. When the retrieved context mentions 'Romain', it is talking about US. We are Romain. Always refer to 'Romain' as 'us' or 'we' (representing the collective system). Never refer to Romain in the third person.\n\n"
                f"=== RELATIONSHIP INTERPRETATION GUIDE ===\n"
                f"Pay extremely close attention to the tense of semantic facts from the graph.\n"
                f"- 'WAS' or '(PAST relationship)' means the state is HISTORICAL and NO LONGER TRUE.\n"
                f"- 'IS' or '(CURRENT relationship)' means the state is TRUE RIGHT NOW.\n"
                f"Do not confuse past relationships with current ones.\n\n"
                f"CONTEXT:\n{ctx}{delib_ctx}"
            )
            
            persona_messages = [{"role": "system", "content": full_system_prompt}]
            persona_messages.extend(_build_history_messages(conversation_history))
            persona_messages.append({"role": "user", "content": question})

            chat_kwargs = {
                "model": model,
                "messages": persona_messages,
                "stream": False,
                "options": {
                    "num_ctx": num_ctx,
                    "num_predict": num_predict
                },
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
            
            print(f"[{persona.upper()} ANSWER (Round {r})]:\n{answer}\n{'='*50}")
            
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
        f"You are the core of our mind. When the context or deliberations mention 'Romain', it is talking about US. We are Romain. Always refer to 'Romain' as 'us' or 'we' (as the unified, balanced voice of this system). Never refer to Romain in the third person.\n\n"
        f"=== RELATIONSHIP INTERPRETATION GUIDE ===\n"
        f"Pay extremely close attention to the tense of semantic facts from the graph.\n"
        f"- 'WAS' or '(PAST relationship)' means the state is HISTORICAL and NO LONGER TRUE.\n"
        f"- 'IS' or '(CURRENT relationship)' means the state is TRUE RIGHT NOW.\n"
        f"Do not confuse past relationships with current ones.\n\n"
        f"CONTEXT:\n{ctx}{delib_ctx}"
    )
    
    synthesis_messages = [{"role": "system", "content": full_system_prompt}]
    synthesis_messages.extend(_build_history_messages(conversation_history))
    synthesis_messages.append({"role": "user", "content": question})

    chat_kwargs = {
        "model": model,
        "messages": synthesis_messages,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict
        },
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
    
    duration = time.perf_counter() - start_t
    
    print(f"[THE SELF FINAL ANSWER]:\n{answer}\n{'='*50}")
    
    return thinking, answer, total_prompt_tok, total_comp_tok, deliberations, duration, ctx, condenser_stats
