"""
ui/personas.py — Persona discovery UI.

Steps:
  0. Configure pipeline parameters.
  1. Load messages & classify emotions → VAD vectors.
     Language comes from the ``"language"`` field in the JSON (set during
     extraction in ``tools/extract_facebook.py``).
  2. Analyse optimal K via Elbow (inertia) and Silhouette score plots.
  3. Cluster with KMeans on VAD vectors (user picks K informed by the analysis).
  4. Explore any cluster (50 sample messages with emotion labels).
  5. Define a persona as a distribution over the K clusters (RPG-style weights
     summing to 1.0).

Alternative: toggle to Semantic (embedding) mode for topic-based clustering.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd


def render_personas_page():
    # Load discovered identities from JSON on first render
    if "discovered_identities" not in st.session_state:
        from tools.persona_clustering import load_discovered_identities
        stored = load_discovered_identities()
        st.session_state["discovered_identities"] = {
            name: info["description"] for name, info in stored.items()
        }

    st.markdown("### :material/groups: Persona Discovery")
    st.caption(
        "Discover emotional personality clusters from your Facebook messages, "
        "then define a persona as a weighted mix of those clusters."
    )

    # ── Step 0: Parameters ─────────────────────────────────────────────────────
    with st.expander("⚙️ Pipeline parameters", expanded=not _has_vectors()):
        col_mode, col_min, col_batch = st.columns([1, 1, 1])
        with col_mode:
            mode = st.selectbox(
                "Clustering mode",
                options=[
                    "Emotion (VAD)",
                    "Direct VAD (BERT)",
                    "LLM VAD (Ollama)",
                    "Semantic (Embedding)",
                ],
                index=0,
                help="**Emotion**: classify emotions → 3D VAD space → emotional clusters.\n\n"
                     "**Direct VAD**: use BERT-VAD model (EmoBank) for direct V/A/D regression "
                     "(no emotion labels, English-centric).\n\n"
                     "**LLM VAD**: prompt a local LLM (Ollama) with conversation context "
                     "to assess Valence/Arousal/Dominance per message.\n\n"
                     "**Semantic**: embed with SentenceTransformer → topic clusters.",
            )
        with col_min:
            min_tokens = st.number_input(
                "Min tokens (words)",
                min_value=1, max_value=50, value=3, step=1,
                help="Ignore messages with fewer words than this (whitespace-split).",
            )
        with col_batch:
            batch_size = st.number_input(
                "Batch size",
                min_value=16, max_value=1024, value=64, step=16,
            )

        if mode == "Emotion (VAD)":
            col_thresh, col_window = st.columns([1, 1])
            with col_thresh:
                neutral_threshold = st.slider(
                    "Neutral threshold",
                    min_value=0.1, max_value=0.8, value=0.4, step=0.05,
                    help="For French model: if max emotion probability < this, "
                         "the message is classified as neutral.",
                )
            with col_window:
                window_size = st.number_input(
                    "Context window",
                    min_value=1, max_value=10, value=1, step=1,
                    key="_window_emo",
                    help="Number of messages to include as context (sliding window "
                         "within each conversation). 1 = no context (classify each "
                         "message independently). E.g. 3 = current message + up to "
                         "2 preceding messages joined with [SEP].",
                )

            # Language opt-out checkboxes
            st.markdown("**Include languages:**")
            col_en_cb, col_fr_cb, _ = st.columns([1, 1, 2])
            with col_en_cb:
                include_en = st.checkbox("🇬🇧 English", value=True, key="_lang_include_en")
            with col_fr_cb:
                include_fr = st.checkbox("🇫🇷 French", value=True, key="_lang_include_fr")
        elif mode == "Direct VAD (BERT)":
            col_window_vad, _ = st.columns([1, 2])
            with col_window_vad:
                window_size = st.number_input(
                    "Context window",
                    min_value=1, max_value=10, value=1, step=1,
                    key="_window_vad",
                    help="Number of messages to include as context (sliding window "
                         "within each conversation). 1 = no context. "
                         "E.g. 3 = current + 2 preceding messages joined with [SEP].",
                )
            neutral_threshold = 0.4
            include_en = True
            include_fr = True
            st.caption(
                "ℹ️ Direct VAD uses `RobroKools/vad-bert` — a BERT model fine-tuned on "
                "EmoBank that outputs continuous Valence, Arousal, Dominance scores. "
                "No emotion labels are produced."
            )
        elif mode == "LLM VAD (Ollama)":
            # ── LLM VAD specific controls ──────────────────────────────────
            neutral_threshold = 0.4
            include_en = True
            include_fr = True

            # Model selector — list available Ollama models
            from config import DEFAULT_OLLAMA
            _ollama_host = st.session_state.get("ollama_host", DEFAULT_OLLAMA)
            _available_models: list[str] = []
            try:
                import ollama as _ollama_mod
                _ollama_client = _ollama_mod.Client(host=_ollama_host)
                _model_list = _ollama_client.list()
                _available_models = sorted(
                    m.get("name", m.get("model", ""))
                    for m in _model_list.get("models", [])
                    if m.get("name") or m.get("model")
                )
            except Exception:
                _available_models = ["llama3:8b", "qwen2.5:7b"]

            col_llm_model, col_llm_window = st.columns([1, 1])
            with col_llm_model:
                llm_model = st.selectbox(
                    "Ollama model",
                    options=_available_models if _available_models else ["llama3:8b"],
                    index=0,
                    key="_llm_vad_model",
                    help="Select the LLM to use for VAD classification via Ollama.",
                )
            with col_llm_window:
                window_size = st.number_input(
                    "Context window",
                    min_value=1, max_value=20, value=5, step=1,
                    key="_window_llm",
                    help="Number of preceding messages (all speakers) to include "
                         "as context for the LLM. E.g. 5 = target message + up to "
                         "5 preceding messages from the conversation.",
                )

            col_smooth, _ = st.columns([1, 2])
            with col_smooth:
                smooth_window = st.number_input(
                    "Smoothing window",
                    min_value=0, max_value=10, value=3, step=1,
                    key="_smooth_llm",
                    help="Rolling-mean smoothing over the VAD sequence "
                         "(0 = no smoothing, 2–3 recommended). "
                         "Applied per-conversation to avoid cross-conversation bleed.",
                )

            # Conversation selector — load available conversations
            if "_llm_available_convs" not in st.session_state:
                try:
                    from tools.persona_clustering import load_conversation_messages
                    _, _, avail = load_conversation_messages(min_tokens=min_tokens)
                    st.session_state["_llm_available_convs"] = avail
                except Exception:
                    st.session_state["_llm_available_convs"] = []

            _avail_convs = st.session_state.get("_llm_available_convs", [])
            if _avail_convs:
                selected_convs = st.multiselect(
                    "Select conversations",
                    options=_avail_convs,
                    default=_avail_convs[:5] if len(_avail_convs) > 5 else _avail_convs,
                    key="_llm_selected_convs",
                    help=f"{len(_avail_convs)} conversations available (≥2 qualifying "
                         "user messages each). Select which ones to analyse.",
                )
            else:
                selected_convs = []
                st.warning("No conversations found. Load Facebook messages first.")

            st.caption(
                "ℹ️ LLM VAD prompts a local LLM via Ollama with conversation context "
                "to assess Valence (-1..1), Arousal (0..1), Dominance (0..1) per message. "
                "First message in each conversation is skipped (no context)."
            )
        else:
            neutral_threshold = 0.4
            window_size = 1
            include_en = True
            include_fr = True

        is_emotion_mode = mode == "Emotion (VAD)"
        is_direct_vad_mode = mode == "Direct VAD (BERT)"
        is_llm_vad_mode = mode == "LLM VAD (Ollama)"

        # ── Buttons: Classify  /  Direct VAD  /  LLM VAD  /  Embed ────────
        if is_emotion_mode:
            classify_label = (
                "🔄 Re-classify Emotions" if _has_vectors()
                else "🧠 Classify Emotions"
            )
            classify_btn = st.button(
                classify_label,
                type="primary",
                use_container_width=True,
            )
        elif is_direct_vad_mode:
            classify_btn = False
            vad_direct_label = (
                "🔄 Re-run VAD" if _has_vectors()
                else "🎯 Run Direct VAD"
            )
            vad_direct_btn = st.button(
                vad_direct_label,
                type="primary",
                use_container_width=True,
            )
        elif is_llm_vad_mode:
            classify_btn = False
            llm_vad_label = (
                "🔄 Re-run LLM VAD" if _has_vectors()
                else "🤖 Run LLM VAD"
            )
            llm_vad_btn = st.button(
                llm_vad_label,
                type="primary",
                use_container_width=True,
                disabled=not selected_convs,  # type: ignore[possibly-undefined]
            )
        else:
            classify_btn = False
            embed_label = "🔄 Re-embed" if _has_vectors() else "🚀 Load & Embed Messages"
            embed_btn = st.button(embed_label, type="primary", use_container_width=True)

    # Build excluded languages set from checkboxes
    excluded_langs: set[str] = set()
    if not include_en:
        excluded_langs.add("en")
    if not include_fr:
        excluded_langs.add("fr")

    # ── Classify Emotions ──────────────────────────────────────────────────
    if is_emotion_mode and classify_btn:
        if len(excluded_langs) >= 2:
            st.error("You must keep at least one language included.")
        else:
            _run_classify(
                min_tokens=min_tokens,
                batch_size=batch_size,
                neutral_threshold=neutral_threshold,
                excluded_langs=excluded_langs,
                window_size=window_size,
            )

    # ── Direct VAD mode: Run BERT-VAD ─────────────────────────────────────
    if is_direct_vad_mode and vad_direct_btn:  # type: ignore[possibly-undefined]
        _run_vad_direct(
            min_tokens=min_tokens,
            batch_size=batch_size,
            window_size=window_size,
        )

    # ── LLM VAD mode: Run LLM classification ──────────────────────────────
    if is_llm_vad_mode and llm_vad_btn:  # type: ignore[possibly-undefined]
        _run_llm_vad(
            min_tokens=min_tokens,
            window_size=window_size,
            model=llm_model,  # type: ignore[possibly-undefined]
            selected_conversations=selected_convs,  # type: ignore[possibly-undefined]
            smooth_window=smooth_window,  # type: ignore[possibly-undefined]
        )

    # ── Semantic mode: Load & Embed ────────────────────────────────────────
    if not is_emotion_mode and not is_direct_vad_mode and not is_llm_vad_mode and embed_btn:  # type: ignore[possibly-undefined]
        _run_embed(min_tokens=min_tokens, batch_size=batch_size)

    if not _has_vectors():
        if is_emotion_mode:
            st.info("Click **Classify Emotions** to start.")
        elif is_direct_vad_mode:
            st.info("Click **Run Direct VAD** to start.")
        elif is_llm_vad_mode:
            st.info("Select conversations and click **Run LLM VAD** to start.")
        else:
            st.info("Configure parameters above and click the button to start.")
        return

    # ── Emotion stats (only in emotion mode) ──────────────────────────────
    if _has_emotion_data():
        _render_emotion_stats()

    # ── VAD stats (direct VAD mode — no emotion labels) ───────────────────
    if not _has_emotion_data() and _has_vad_data():
        _render_vad_stats()

    # ── LLM VAD trajectory (when speaker data is available) ───────────────
    if st.session_state.get("persona_speakers") and _has_vad_data():
        _render_llm_vad_trajectory()

    # ── Step 1: Find optimal K ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Find Optimal K")
    st.caption(
        "Run the **Elbow method** (inertia) and **Silhouette score** across a range of K "
        "to find the best number of clusters before committing."
    )

    col_kmin, col_kmax, col_run = st.columns([1, 1, 2])
    with col_kmin:
        k_min = st.number_input("K min", min_value=2, max_value=30, value=2, step=1)
    with col_kmax:
        k_max = st.number_input("K max", min_value=3, max_value=40, value=20, step=1)
    with col_run:
        st.write("")  # spacer
        analyse_btn = st.button(
            "🔍 Analyse K range" if not _has_k_analysis() else "🔄 Re-analyse",
            use_container_width=True,
        )

    if analyse_btn:
        _run_k_analysis(k_min=k_min, k_max=k_max)

    if _has_k_analysis():
        analysis = st.session_state["persona_k_analysis"]
        _render_k_charts(analysis)

    # ── Step 2: Cluster ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎯 Cluster Messages")

    default_k = st.session_state.get("persona_k_analysis", {}).get("best_k_silhouette", 7)
    col_k, col_cluster = st.columns([1, 2])
    with col_k:
        k = st.number_input(
            "Number of clusters (K)",
            min_value=2, max_value=30, value=default_k, step=1,
            help="Pick K informed by the analysis above.",
        )
    with col_cluster:
        st.write("")  # spacer
        cluster_btn = st.button(
            "🚀 Run Clustering" if not _has_clusters() else "🔄 Re-cluster",
            type="primary",
            use_container_width=True,
        )

    if cluster_btn:
        _run_clustering(k=k)

    if not _has_clusters():
        st.info("Pick K above and click **Run Clustering**.")
        return

    # ── Cluster overview ──────────────────────────────────────────────────
    stats: list[dict] = st.session_state["persona_cluster_stats"]
    k_actual = len(stats)

    st.markdown("---")
    st.markdown("#### Cluster Overview")

    # Bar chart
    df_stats = pd.DataFrame(stats)
    df_stats["label"] = df_stats["cluster"].apply(
        lambda c: st.session_state.get(f"persona_cluster_name_{c}", f"Cluster {c}")
    )
    chart_df = df_stats.set_index("label")["count"]
    st.bar_chart(chart_df, color="#6366f1")

    # Stats table with emotion info if available
    if _has_emotion_data():
        _render_cluster_emotion_table(stats, k_actual)
    elif _has_vad_data():
        _render_cluster_vad_table(stats, k_actual)
    else:
        st.dataframe(
            df_stats[["cluster", "label", "count", "pct"]].rename(
                columns={"cluster": "#", "label": "Name", "count": "Messages", "pct": "%"}
            ),
            width="stretch",
            hide_index=True,
        )

    # 3D VAD scatter plot (when VAD vectors are available)
    if _has_vad_data():
        _render_vad_scatter()

    # ── Step 3: Explore a cluster ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Explore Cluster")

    col_sel, col_name = st.columns([1, 2])
    with col_sel:
        cluster_id = st.selectbox(
            "Select cluster",
            options=list(range(k_actual)),
            format_func=lambda c: (
                f"Cluster {c} — "
                + st.session_state.get(f"persona_cluster_name_{c}", "unnamed")
                + f" ({stats[c]['count']} msgs)"
            ),
        )
    with col_name:
        current_name = st.session_state.get(f"persona_cluster_name_{cluster_id}", f"Cluster {cluster_id}")
        new_name = st.text_input(
            "Rename cluster",
            value=current_name,
            key=f"_rename_cluster_{cluster_id}",
        )
        if new_name != current_name:
            st.session_state[f"persona_cluster_name_{cluster_id}"] = new_name
            st.rerun()

    # Show sample messages
    from tools.persona_clustering import cluster_sample

    samples = cluster_sample(
        st.session_state["persona_messages"],
        st.session_state["persona_labels"],
        cluster_id,
        n=50,
    )

    cluster_label = st.session_state.get(f"persona_cluster_name_{cluster_id}", f"Cluster {cluster_id}")
    emotion_probs = st.session_state.get("persona_emotion_probs")

    with st.expander(
        f"📝 {len(samples)} sample messages from *{cluster_label}*",
        expanded=False,
    ):
        _msg_cards = []
        emoji_map = {
            "joy": "😊", "anger": "😠", "fear": "😨",
            "sadness": "😢", "neutral": "😐",
        }
        for i, msg in enumerate(samples):
            date_str = msg.get("date", "")[:10]
            conv = msg.get("conversation", "")
            text = msg.get("text", "").replace("<", "<").replace(">", ">")

            # Emotion / VAD badge if available
            emotion_badge = ""
            if emotion_probs and msg.get("index") is not None:
                idx = msg["index"]
                if idx < len(emotion_probs):
                    from tools.persona_clustering import dominant_emotion
                    probs = emotion_probs[idx]
                    dom = dominant_emotion(probs)
                    vad = st.session_state.get("persona_vad_vectors")
                    vad_str = ""
                    if vad is not None and idx < len(vad):
                        v, a, d = vad[idx]
                        vad_str = f" · V={v:.2f} A={a:.2f} D={d:.2f}"
                    emotion_badge = (
                        f" <span style='padding:2px 6px;"
                        f"border:1px solid rgba(128,128,128,0.3);"
                        f"border-radius:4px;font-size:0.85em'>"
                        f"{emoji_map.get(dom, '❓')}{dom}{vad_str}</span>"
                    )
            elif not emotion_probs and msg.get("index") is not None:
                # Direct VAD mode — show V/A/D scores without emotion label
                idx = msg["index"]
                vad = st.session_state.get("persona_vad_vectors")
                if vad is not None and idx < len(vad):
                    v, a, d = vad[idx]
                    emotion_badge = (
                        f" <span style='padding:2px 6px;"
                        f"border:1px solid rgba(128,128,128,0.3);"
                        f"border-radius:4px;font-size:0.85em'>"
                        f"V={v:.2f} A={a:.2f} D={d:.2f}</span>"
                    )

            _msg_cards.append(
                f"<div style='padding:6px 10px;margin:3px 0;"
                f"border:1px solid rgba(128,128,128,0.25);"
                f"border-radius:6px;font-size:0.85em'>"
                f"<b>#{i+1}</b> <span style='color:gray'>{date_str}</span>"
                f" · <code>{conv[:30]}</code>"
                f"{emotion_badge}"
                f" — {text[:300]}</div>"
            )

        st.markdown(
            "<div style='max-height:500px;overflow-y:auto;"
            "padding-right:6px'>"
            + "\n".join(_msg_cards)
            + "</div>",
            unsafe_allow_html=True,
        )

    # ── Step 3.5: Sub-Cluster Analysis ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🧬 Sub-Cluster Analysis")
    st.caption(
        "Run sub-clustering within the selected cluster to discover "
        "fine-grained personality facets. Then use LLM to generate "
        "personality descriptions."
    )

    col_subk, col_subrun = st.columns([1, 2])
    with col_subk:
        sub_k = st.number_input(
            "Sub-clusters (K)",
            min_value=2, max_value=6, value=3, step=1,
            key="_sub_k",
            help="Number of sub-clusters within the selected cluster.",
        )
    with col_subrun:
        st.write("")  # spacer
        sub_cluster_btn = st.button(
            "🔬 Run Sub-Clustering",
            use_container_width=True,
            key="_run_sub_cluster",
        )

    if sub_cluster_btn:
        _run_sub_clustering(cluster_id, sub_k)

    # Show sub-cluster results if available
    sub_key = f"persona_sub_labels_{cluster_id}"
    if sub_key in st.session_state and st.session_state[sub_key] is not None:
        _render_sub_clusters(cluster_id)

    # ── Step 4: Define persona distribution ───────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎲 Define Persona Distribution")
    st.caption(
        "Set a weight for each cluster (like RPG stats). "
        "The weights must sum to **1.0**."
    )

    # Initialize weights in session_state (both the list and per-slider keys)
    if "persona_weights" not in st.session_state or len(st.session_state["persona_weights"]) != k_actual:
        default_w = round(1.0 / k_actual, 3)
        st.session_state["persona_weights"] = [default_w] * k_actual
        for i in range(k_actual):
            st.session_state[f"_persona_w_{i}"] = default_w

    # Render sliders in columns (2 per row)
    # Sliders are driven entirely by their session_state key — no explicit
    # ``value=`` parameter, which would conflict after auto-normalize sets
    # the key and calls ``st.rerun()``.
    cols_per_row = 2
    for row_start in range(0, k_actual, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= k_actual:
                break
            with col:
                label = st.session_state.get(f"persona_cluster_name_{idx}", f"Cluster {idx}")
                # Ensure the widget key exists (e.g. after re-cluster with new K)
                if f"_persona_w_{idx}" not in st.session_state:
                    st.session_state[f"_persona_w_{idx}"] = float(
                        st.session_state["persona_weights"][idx]
                    )
                st.slider(
                    f"**{label}** (cluster {idx})",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    key=f"_persona_w_{idx}",
                )

    # Read current weights back from slider widget keys
    weights = [
        float(st.session_state.get(f"_persona_w_{i}", round(1.0 / k_actual, 3)))
        for i in range(k_actual)
    ]
    total = sum(weights)
    st.session_state["persona_weights"] = weights

    # Visual feedback
    col_total, col_action = st.columns([2, 1])
    with col_total:
        if abs(total - 1.0) <= 0.01:
            st.success(f"✅ Total = {total:.2f} — valid distribution")
        else:
            st.warning(f"⚠️ Total = {total:.2f} — must equal 1.0 (off by {total - 1.0:+.2f})")

    with col_action:
        if st.button("⚖️ Auto-normalize", use_container_width=True):
            if total > 0:
                normalized = [round(w / total, 3) for w in weights]
                st.session_state["persona_weights"] = normalized
                # Also update the slider widget keys so they pick up
                # the new values on rerun (widgets read from their key).
                for i, nw in enumerate(normalized):
                    st.session_state[f"_persona_w_{i}"] = nw
                st.rerun()

    # Show distribution as a horizontal stacked bar
    if abs(total - 1.0) <= 0.01:
        dist_df = pd.DataFrame({
            st.session_state.get(f"persona_cluster_name_{i}", f"Cluster {i}"): [weights[i]]
            for i in range(k_actual)
        })
        st.bar_chart(dist_df.T, horizontal=True, color="#a78bfa")

    # Save button
    st.markdown("---")
    persona_name = st.text_input("Persona name", value="My Persona", key="_persona_name")
    if st.button("💾 Save Persona", type="primary", use_container_width=True):
        if abs(total - 1.0) > 0.01:
            st.error("Weights must sum to 1.0 before saving.")
        else:
            _save_persona(persona_name, weights, k_actual)
            st.success(f"Persona **{persona_name}** saved!")

    # Show saved personas
    if st.session_state.get("saved_personas"):
        st.markdown("---")
        st.markdown("#### 📋 Saved Personas")
        for p in st.session_state["saved_personas"]:
            with st.expander(f"**{p['name']}**"):
                for i, w in enumerate(p["weights"]):
                    label = st.session_state.get(f"persona_cluster_name_{i}", f"Cluster {i}")
                    st.write(f"- {label}: **{w:.2f}**")


# ── internal helpers ───────────────────────────────────────────────────────────

def _has_vectors() -> bool:
    return "persona_embeddings" in st.session_state and st.session_state["persona_embeddings"] is not None


def _has_emotion_data() -> bool:
    return "persona_emotion_probs" in st.session_state and st.session_state["persona_emotion_probs"] is not None


def _has_vad_data() -> bool:
    return "persona_vad_vectors" in st.session_state and st.session_state["persona_vad_vectors"] is not None


def _has_k_analysis() -> bool:
    return "persona_k_analysis" in st.session_state and st.session_state["persona_k_analysis"] is not None


def _has_clusters() -> bool:
    return "persona_labels" in st.session_state and st.session_state["persona_labels"] is not None


def _run_classify(
    min_tokens: int,
    batch_size: int,
    neutral_threshold: float,
    excluded_langs: set[str] | None = None,
    window_size: int = 1,
):
    """Load messages, filter by language, classify emotions, and map to VAD.

    Language is read from the ``"language"`` field in each message dict
    (set during extraction by ``tools/extract_facebook.py``).  Messages
    whose language is in *excluded_langs* are dropped before classification.

    When *window_size* > 1, a conversation-aware sliding window is used:
    preceding messages from the same conversation are concatenated with
    ``[SEP]`` to give the emotion classifier conversational context.
    """
    from tools.persona_clustering import (
        load_my_messages, classify_emotions, emotions_to_vad,
    )

    if excluded_langs is None:
        excluded_langs = set()

    # Step 1: Load messages
    with st.spinner("Loading Facebook messages…"):
        messages, self_name = load_my_messages(min_tokens=min_tokens)
        st.session_state["persona_self_name"] = self_name

    st.toast(f"Loaded **{len(messages):,}** messages from **{self_name}**", icon="📨")

    # Step 2: Read language from each message (defaults to "en" if missing)
    languages = [m.get("language", "en") for m in messages]

    # Filter out excluded languages
    if excluded_langs:
        keep_mask = [lang not in excluded_langs for lang in languages]
        removed = sum(1 for k in keep_mask if not k)
        messages = [m for m, k in zip(messages, keep_mask) if k]
        languages = [l for l, k in zip(languages, keep_mask) if k]
        st.toast(
            f"Excluded {removed:,} messages in {', '.join(sorted(excluded_langs)).upper()}",
            icon="🚫",
        )

    st.session_state["persona_messages"] = messages
    st.session_state["persona_languages"] = languages

    en_count = sum(1 for l in languages if l == "en")
    fr_count = sum(1 for l in languages if l == "fr")
    st.session_state["persona_lang_stats"] = {
        "en": en_count, "fr": fr_count,
    }

    texts = [m["text"] for m in messages]

    # Step 3: Classify emotions
    progress = st.progress(0, text="Classifying emotions…")

    def _emo_cb(current: int, total: int):
        progress.progress(current / total, text=f"Classifying emotions… {current:,}/{total:,}")

    emotion_probs = classify_emotions(
        texts,
        languages=languages,
        batch_size=batch_size,
        neutral_threshold=neutral_threshold,
        progress_callback=_emo_cb,
        messages=messages,
        window_size=window_size,
    )
    progress.empty()
    st.session_state["persona_emotion_probs"] = emotion_probs

    # Step 4: Map to VAD vectors
    with st.spinner("Mapping to VAD space…"):
        vad_vectors = emotions_to_vad(emotion_probs)
        st.session_state["persona_vad_vectors"] = vad_vectors
        # Store as "embeddings" for downstream compat (find_optimal_k, cluster_embeddings)
        st.session_state["persona_embeddings"] = vad_vectors

    # Clear downstream state
    st.session_state.pop("persona_k_analysis", None)
    st.session_state.pop("persona_labels", None)
    st.session_state.pop("persona_cluster_stats", None)

    st.toast(f"Classified {len(texts):,} messages → {vad_vectors.shape[1]}D VAD vectors", icon="🧠")


def _run_vad_direct(
    min_tokens: int,
    batch_size: int,
    window_size: int = 1,
):
    """Load messages and classify directly into VAD space using BERT-VAD."""
    from tools.persona_clustering import load_my_messages, classify_vad_direct

    # Step 1: Load messages
    with st.spinner("Loading Facebook messages…"):
        messages, self_name = load_my_messages(min_tokens=min_tokens)
        st.session_state["persona_self_name"] = self_name

    st.toast(f"Loaded **{len(messages):,}** messages from **{self_name}**", icon="📨")

    st.session_state["persona_messages"] = messages

    texts = [m["text"] for m in messages]

    # Step 2: Direct VAD classification
    progress = st.progress(0, text="Running BERT-VAD…")

    def _vad_cb(current: int, total: int):
        progress.progress(current / total, text=f"BERT-VAD… {current:,}/{total:,}")

    vad_vectors = classify_vad_direct(
        texts,
        messages=messages,
        window_size=window_size,
        batch_size=batch_size,
        progress_callback=_vad_cb,
    )
    progress.empty()

    st.session_state["persona_vad_vectors"] = vad_vectors
    # Store as "embeddings" for downstream compat (find_optimal_k, cluster_embeddings)
    st.session_state["persona_embeddings"] = vad_vectors

    # Clear emotion-specific state (direct VAD has no emotion labels)
    st.session_state.pop("persona_emotion_probs", None)
    st.session_state.pop("persona_languages", None)
    st.session_state.pop("persona_lang_stats", None)

    # Clear downstream state
    st.session_state.pop("persona_k_analysis", None)
    st.session_state.pop("persona_labels", None)
    st.session_state.pop("persona_cluster_stats", None)

    st.toast(f"Classified {len(texts):,} messages → 3D VAD vectors (direct)", icon="🎯")


def _run_llm_vad(
    min_tokens: int,
    window_size: int,
    model: str,
    selected_conversations: list[str],
    smooth_window: int = 3,
):
    """Load conversation messages and classify into VAD space using an LLM via Ollama."""
    from tools.persona_clustering import load_conversation_messages, classify_vad_llm
    from config import DEFAULT_OLLAMA

    ollama_host = st.session_state.get("ollama_host", DEFAULT_OLLAMA)

    # Step 1: Load all messages from selected conversations
    with st.spinner("Loading conversation messages…"):
        all_messages, self_name, _ = load_conversation_messages(
            min_tokens=min_tokens,
            selected_conversations=selected_conversations,
        )
        st.session_state["persona_self_name"] = self_name

    if not all_messages:
        st.error("No messages found in the selected conversations.")
        return

    st.toast(
        f"Loaded **{len(all_messages):,}** messages (all speakers) from "
        f"**{len(selected_conversations)}** conversations",
        icon="📨",
    )

    # Step 2: LLM VAD classification with progress bar
    progress = st.progress(0, text="Running LLM VAD classification…")

    def _llm_cb(current: int, total: int):
        pct = current / total if total > 0 else 0
        progress.progress(pct, text=f"LLM VAD… {current:,}/{total:,} messages")

    classified_messages, vad_vectors, vad_deltas, vad_smoothed = classify_vad_llm(
        all_messages=all_messages,
        self_name=self_name,
        window_size=window_size,
        min_tokens=min_tokens,
        model=model,
        ollama_host=ollama_host,
        progress_callback=_llm_cb,
        smooth_window=smooth_window,
    )
    progress.empty()

    if len(classified_messages) == 0:
        st.error(
            "No messages were classified. Check that the selected conversations "
            "have ≥2 messages each."
        )
        return

    # Store results — classified_messages include ALL speakers
    st.session_state["persona_messages"] = classified_messages
    st.session_state["persona_vad_vectors"] = vad_vectors
    st.session_state["persona_vad_deltas"] = vad_deltas
    st.session_state["persona_vad_smoothed"] = vad_smoothed
    # Store speaker names for trajectory coloring
    st.session_state["persona_speakers"] = [
        m.get("speaker", m.get("sender_name", "Unknown"))
        for m in classified_messages
    ]
    # Store as "embeddings" for downstream compat (find_optimal_k, cluster_embeddings)
    st.session_state["persona_embeddings"] = vad_vectors

    # Clear emotion-specific state (LLM VAD has no emotion labels)
    st.session_state.pop("persona_emotion_probs", None)
    st.session_state.pop("persona_languages", None)
    st.session_state.pop("persona_lang_stats", None)

    # Clear downstream state
    st.session_state.pop("persona_k_analysis", None)
    st.session_state.pop("persona_labels", None)
    st.session_state.pop("persona_cluster_stats", None)

    st.toast(
        f"Classified {len(classified_messages):,} messages (all speakers) → "
        f"3D VAD vectors (LLM: {model})",
        icon="🤖",
    )


def _run_embed(min_tokens: int, batch_size: int):
    """Load messages and embed them (semantic / topic mode)."""
    from tools.persona_clustering import load_my_messages, embed_messages

    # Step 1: Load messages
    with st.spinner("Loading Facebook messages…"):
        messages, self_name = load_my_messages(min_tokens=min_tokens)
        st.session_state["persona_messages"] = messages
        st.session_state["persona_self_name"] = self_name

    st.toast(f"Loaded **{len(messages):,}** messages from **{self_name}**", icon="📨")

    # Step 2: Embed
    texts = [m["text"] for m in messages]
    progress = st.progress(0, text="Embedding messages…")

    def _progress_cb(current: int, total: int):
        progress.progress(current / total, text=f"Embedding messages… {current:,}/{total:,}")

    embeddings = embed_messages(texts, batch_size=batch_size, progress_callback=_progress_cb)
    progress.empty()
    st.session_state["persona_embeddings"] = embeddings

    # Clear emotion-specific state
    st.session_state.pop("persona_emotion_probs", None)
    st.session_state.pop("persona_vad_vectors", None)
    st.session_state.pop("persona_languages", None)
    st.session_state.pop("persona_lang_stats", None)

    # Clear downstream state
    st.session_state.pop("persona_k_analysis", None)
    st.session_state.pop("persona_labels", None)
    st.session_state.pop("persona_cluster_stats", None)

    st.toast(f"Embedded {len(texts):,} messages → {embeddings.shape[1]}d vectors", icon="🧮")


def _render_emotion_stats():
    """Render emotion distribution charts after classification."""
    from tools.persona_clustering import EMOTION_LABELS

    st.markdown("---")
    st.markdown("#### 🎭 Emotion Distribution")

    emotion_probs = st.session_state["persona_emotion_probs"]
    lang_stats = st.session_state.get("persona_lang_stats", {})

    # Language stats summary
    en_count = lang_stats.get("en", 0)
    fr_count = lang_stats.get("fr", 0)
    total_msgs = en_count + fr_count
    en_convs = lang_stats.get("en_convs", 0)
    fr_convs = lang_stats.get("fr_convs", 0)
    total_convs = en_convs + fr_convs
    if total_msgs > 0:
        st.caption(
            f"**{total_msgs:,}** messages classified · "
            f"🇬🇧 {en_count:,} English ({en_count/total_msgs*100:.1f}%) · "
            f"🇫🇷 {fr_count:,} French ({fr_count/total_msgs*100:.1f}%) · "
            f"**{total_convs}** conversations ({en_convs} EN / {fr_convs} FR)"
        )

    # Average emotion probabilities
    avg_probs = {}
    for emotion in EMOTION_LABELS:
        avg_probs[emotion] = np.mean([p.get(emotion, 0.0) for p in emotion_probs])

    # Dominant emotion counts
    from tools.persona_clustering import dominant_emotion
    dom_counts = {}
    for p in emotion_probs:
        dom = dominant_emotion(p)
        dom_counts[dom] = dom_counts.get(dom, 0) + 1

    col_avg, col_dom = st.columns(2)

    with col_avg:
        st.markdown("##### Average Probabilities")
        emoji_map = {"joy": "😊", "anger": "😠", "fear": "😨", "sadness": "😢", "neutral": "😐"}
        df_avg = pd.DataFrame({
            "Emotion": [f"{emoji_map.get(e, '')} {e}" for e in EMOTION_LABELS],
            "Avg Probability": [avg_probs[e] for e in EMOTION_LABELS],
        }).set_index("Emotion")
        st.bar_chart(df_avg, color="#6366f1")

    with col_dom:
        st.markdown("##### Dominant Emotion Counts")
        df_dom = pd.DataFrame({
            "Emotion": [f"{emoji_map.get(e, '')} {e}" for e in EMOTION_LABELS],
            "Count": [dom_counts.get(e, 0) for e in EMOTION_LABELS],
        }).set_index("Emotion")
        st.bar_chart(df_dom, color="#22c55e")

    # Global VAD Dominance variance
    vad = st.session_state.get("persona_vad_vectors")
    if vad is not None:
        d_mean = float(np.mean(vad[:, 2]))
        d_var = float(np.var(vad[:, 2]))
        d_std = float(np.std(vad[:, 2]))

        st.markdown("##### Dominance Variance")
        col_d_mean, col_d_var, col_d_std = st.columns(3)
        col_d_mean.metric("Mean Dominance", f"{d_mean:.4f}")
        col_d_var.metric("Variance", f"{d_var:.4f}")
        col_d_std.metric("Std Dev", f"{d_std:.4f}")
        st.caption(
            "Dominance ranges from 0 (submissive) to 1 (dominant). "
            "Higher variance indicates more emotional range between "
            "submissive and dominant states across your messages."
        )


def _render_vad_stats():
    """Render VAD distribution stats for direct VAD mode (no emotion labels)."""
    vad = st.session_state.get("persona_vad_vectors")
    if vad is None:
        return

    st.markdown("---")
    st.markdown("#### 🎯 VAD Distribution (Direct)")
    st.caption(
        f"**{len(vad):,}** messages classified with BERT-VAD (direct regression)."
    )

    # Summary metrics
    col_v, col_a, col_d = st.columns(3)
    with col_v:
        st.metric("Valence (mean)", f"{vad[:, 0].mean():.4f}")
        st.caption(f"std={vad[:, 0].std():.4f}  range=[{vad[:, 0].min():.3f}, {vad[:, 0].max():.3f}]")
    with col_a:
        st.metric("Arousal (mean)", f"{vad[:, 1].mean():.4f}")
        st.caption(f"std={vad[:, 1].std():.4f}  range=[{vad[:, 1].min():.3f}, {vad[:, 1].max():.3f}]")
    with col_d:
        st.metric("Dominance (mean)", f"{vad[:, 2].mean():.4f}")
        st.caption(f"std={vad[:, 2].std():.4f}  range=[{vad[:, 2].min():.3f}, {vad[:, 2].max():.3f}]")

    # Variance metrics
    st.markdown("##### Variance")
    col_vv, col_av, col_dv = st.columns(3)
    col_vv.metric("Valence Var", f"{vad[:, 0].var():.4f}")
    col_av.metric("Arousal Var", f"{vad[:, 1].var():.4f}")
    col_dv.metric("Dominance Var", f"{vad[:, 2].var():.4f}")

    # Histograms
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.markdown("##### Valence")
        _vc_v = pd.DataFrame({"Valence": vad[:, 0]})["Valence"].value_counts(bins=20).sort_index()
        _vc_v.index = [f"{iv.mid:.2f}" for iv in _vc_v.index]
        st.bar_chart(_vc_v, color="#6366f1")
    with col_h2:
        st.markdown("##### Arousal")
        _vc_a = pd.DataFrame({"Arousal": vad[:, 1]})["Arousal"].value_counts(bins=20).sort_index()
        _vc_a.index = [f"{iv.mid:.2f}" for iv in _vc_a.index]
        st.bar_chart(_vc_a, color="#22c55e")
    with col_h3:
        st.markdown("##### Dominance")
        _vc_d = pd.DataFrame({"Dominance": vad[:, 2]})["Dominance"].value_counts(bins=20).sort_index()
        _vc_d.index = [f"{iv.mid:.2f}" for iv in _vc_d.index]
        st.bar_chart(_vc_d, color="#f59e0b")


def _render_cluster_emotion_table(stats: list[dict], k_actual: int):
    """Render cluster stats table with per-cluster emotion breakdown."""
    from tools.persona_clustering import EMOTION_LABELS, dominant_emotion

    emotion_probs = st.session_state.get("persona_emotion_probs", [])
    labels = st.session_state.get("persona_labels")
    emoji_map = {"joy": "😊", "anger": "😠", "fear": "😨", "sadness": "😢", "neutral": "😐"}

    rows = []
    for s in stats:
        c = s["cluster"]
        row = {
            "#": c,
            "Name": st.session_state.get(f"persona_cluster_name_{c}", f"Cluster {c}"),
            "Messages": s["count"],
            "%": s["pct"],
        }

        if labels is not None and emotion_probs:
            # Get indices for this cluster
            cluster_indices = np.where(labels == c)[0]
            if len(cluster_indices) > 0:
                # Average emotion probs for this cluster
                cluster_probs = [emotion_probs[i] for i in cluster_indices]
                avg = {e: np.mean([p.get(e, 0.0) for p in cluster_probs]) for e in EMOTION_LABELS}
                dom = max(avg, key=avg.get)  # type: ignore[arg-type]
                row["Dominant"] = f"{emoji_map.get(dom, '')} {dom}"

                # Average VAD
                vad = st.session_state.get("persona_vad_vectors")
                if vad is not None:
                    cluster_vad = vad[cluster_indices]
                    row["V"] = round(float(np.mean(cluster_vad[:, 0])), 3)
                    row["A"] = round(float(np.mean(cluster_vad[:, 1])), 3)
                    row["D"] = round(float(np.mean(cluster_vad[:, 2])), 3)

        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def _render_cluster_vad_table(stats: list[dict], k_actual: int):
    """Render cluster stats table with per-cluster VAD means (no emotion labels)."""
    vad = st.session_state.get("persona_vad_vectors")
    labels = st.session_state.get("persona_labels")

    rows = []
    for s in stats:
        c = s["cluster"]
        row = {
            "#": c,
            "Name": st.session_state.get(f"persona_cluster_name_{c}", f"Cluster {c}"),
            "Messages": s["count"],
            "%": s["pct"],
        }

        if vad is not None and labels is not None:
            cluster_indices = np.where(labels == c)[0]
            if len(cluster_indices) > 0:
                cluster_vad = vad[cluster_indices]
                row["V̄"] = round(float(np.mean(cluster_vad[:, 0])), 3)
                row["Ā"] = round(float(np.mean(cluster_vad[:, 1])), 3)
                row["D̄"] = round(float(np.mean(cluster_vad[:, 2])), 3)
                row["V σ"] = round(float(np.std(cluster_vad[:, 0])), 3)
                row["A σ"] = round(float(np.std(cluster_vad[:, 1])), 3)
                row["D σ"] = round(float(np.std(cluster_vad[:, 2])), 3)

        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def _render_vad_scatter():
    """Render interactive 3D VAD scatter plot colored by cluster."""
    import plotly.express as px

    vad = st.session_state.get("persona_vad_vectors")
    labels = st.session_state.get("persona_labels")

    if vad is None or labels is None:
        return

    st.markdown("---")
    st.markdown("#### 🌐 3D VAD Space")
    st.caption("Interactive 3D scatter plot of messages in Valence-Arousal-Dominance space, colored by cluster.")

    # Subsample for performance (max 5000 points)
    n = len(vad)
    max_points = 5000
    if n > max_points:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n, size=max_points, replace=False)
    else:
        sample_idx = np.arange(n)

    df_scatter = pd.DataFrame({
        "Valence": vad[sample_idx, 0],
        "Arousal": vad[sample_idx, 1],
        "Dominance": vad[sample_idx, 2],
        "Cluster": [
            st.session_state.get(f"persona_cluster_name_{labels[i]}", f"Cluster {labels[i]}")
            for i in sample_idx
        ],
    })

    fig = px.scatter_3d(
        df_scatter,
        x="Valence", y="Dominance", z="Arousal",
        color="Cluster",
        opacity=0.5,
        height=600,
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Valence (negative ← → positive)",
            yaxis_title="Dominance (submissive ← → dominant)",
            zaxis_title="Arousal (calm ← → excited)",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, width="stretch")


def _render_llm_vad_trajectory():
    """Render 3D VAD trajectory plot with chronological lines, color-coded by speaker.

    Shows the emotional trajectory through VAD space over time, with:
    - Points connected in chronological order per conversation
    - Color-coding by speaker
    - Optional smoothed overlay
    - Delta statistics table
    """
    import plotly.graph_objects as go

    vad = st.session_state.get("persona_vad_vectors")
    speakers = st.session_state.get("persona_speakers")
    messages = st.session_state.get("persona_messages")
    vad_deltas = st.session_state.get("persona_vad_deltas")
    vad_smoothed = st.session_state.get("persona_vad_smoothed")

    if vad is None or speakers is None or messages is None:
        return

    st.markdown("---")
    st.markdown("#### 🧭 Emotional Trajectory")
    st.caption(
        "3D trajectory through Valence-Arousal-Dominance space, "
        "connecting messages in chronological order per conversation. "
        "Color-coded by speaker."
    )

    n = len(vad)

    # Group messages by conversation, preserving order
    conv_groups: dict[str, list[int]] = {}
    for i, m in enumerate(messages):
        conv = m.get("conversation", "unknown")
        if conv not in conv_groups:
            conv_groups[conv] = []
        conv_groups[conv].append(i)

    # Unique speakers → color map
    unique_speakers = sorted(set(speakers))
    # Use a qualitative color palette
    _colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]
    speaker_colors = {
        s: _colors[i % len(_colors)] for i, s in enumerate(unique_speakers)
    }

    fig = go.Figure()

    # Use smoothed data if available, raw otherwise
    use_vad = vad_smoothed if vad_smoothed is not None else vad

    # Track which speakers already have a legend entry
    _legend_shown: set[str] = set()

    # For each conversation, draw per-speaker lines+markers
    for conv, indices in conv_groups.items():
        conv_short = conv[:25] + "…" if len(conv) > 25 else conv

        # Draw per-speaker lines connecting their points chronologically
        for speaker in unique_speakers:
            speaker_indices = [i for i in indices if speakers[i] == speaker]
            if not speaker_indices:
                continue

            hover_texts = []
            for idx in speaker_indices:
                m = messages[idx]
                text_preview = m.get("text", "")[:80]
                date = m.get("date", "")[:10]
                v, a, d = vad[idx]
                delta_str = ""
                if vad_deltas is not None:
                    dv, da, dd = vad_deltas[idx]
                    delta_str = f"<br>ΔV={dv:+.2f} ΔA={da:+.2f} ΔD={dd:+.2f}"
                hover_texts.append(
                    f"<b>[{speaker}]</b> {date} · {conv_short}<br>"
                    f"V={v:.2f} A={a:.2f} D={d:.2f}{delta_str}<br>"
                    f"{text_preview}"
                )

            show_legend = speaker not in _legend_shown
            _legend_shown.add(speaker)

            fig.add_trace(go.Scatter3d(
                x=use_vad[speaker_indices, 0],
                y=use_vad[speaker_indices, 2],  # Dominance
                z=use_vad[speaker_indices, 1],  # Arousal
                mode="lines+markers",
                line=dict(
                    color=speaker_colors[speaker],
                    width=3,
                ),
                marker=dict(
                    size=4,
                    color=speaker_colors[speaker],
                    opacity=0.9,
                ),
                name=speaker,
                text=hover_texts,
                hoverinfo="text",
                legendgroup=speaker,
                showlegend=show_legend,
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Valence (negative ← → positive)",
            yaxis_title="Dominance (submissive ← → dominant)",
            zaxis_title="Arousal (calm ← → excited)",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=700,
        legend=dict(
            title="Speaker",
            itemsizing="constant",
        ),
    )
    st.plotly_chart(fig, width="stretch")

    # ── Delta statistics ───────────────────────────────────────────────────
    if vad_deltas is not None and len(vad_deltas) > 0:
        st.markdown("##### 📈 VAD Delta Statistics")
        st.caption(
            "Change between consecutive messages. Large deltas indicate "
            "emotional shifts in the conversation."
        )

        # Per-speaker delta stats
        delta_rows = []
        for speaker in unique_speakers:
            sp_mask = [i for i, s in enumerate(speakers) if s == speaker]
            if len(sp_mask) < 2:
                continue
            sp_deltas = vad_deltas[sp_mask]
            sp_vad = vad[sp_mask]

            # Magnitude of each delta vector (Euclidean norm)
            delta_magnitudes = np.linalg.norm(sp_deltas, axis=1)
            # Cumulative emotional drift (total distance traveled in VAD space)
            cumulative_drift = float(delta_magnitudes.sum())
            # Emotional volatility (std of VAD values per dimension)
            vol_v = float(np.std(sp_vad[:, 0]))
            vol_a = float(np.std(sp_vad[:, 1]))
            vol_d = float(np.std(sp_vad[:, 2]))

            delta_rows.append({
                "Speaker": speaker,
                "Msgs": len(sp_mask),
                "|Δ| mean": round(float(delta_magnitudes.mean()), 3),
                "|Δ| max": round(float(delta_magnitudes.max()), 3),
                "Drift": round(cumulative_drift, 2),
                "σ(V)": round(vol_v, 3),
                "σ(A)": round(vol_a, 3),
                "σ(D)": round(vol_d, 3),
                "V̄": round(float(sp_vad[:, 0].mean()), 3),
                "Ā": round(float(sp_vad[:, 1].mean()), 3),
                "D̄": round(float(sp_vad[:, 2].mean()), 3),
            })

        if delta_rows:
            df_deltas = pd.DataFrame(delta_rows)
            st.dataframe(df_deltas, width="stretch", hide_index=True)
            st.caption(
                "**|Δ|** = Euclidean magnitude of delta vector · "
                "**Drift** = cumulative distance traveled in VAD space · "
                "**σ** = emotional volatility (std of VAD values)"
            )

    # Show smoothing info
    if vad_smoothed is not None:
        st.caption("ℹ️ Trajectory uses **smoothed** VAD values (rolling mean).")
    else:
        st.caption("ℹ️ Trajectory uses **raw** VAD values (no smoothing).")


def _run_k_analysis(k_min: int, k_max: int):
    """Run Elbow + Silhouette analysis over a K range."""
    from tools.persona_clustering import find_optimal_k

    embeddings = st.session_state["persona_embeddings"]
    k_range = range(k_min, k_max + 1)

    progress = st.progress(0, text="Analysing K range…")

    def _progress_cb(current: int, total: int):
        progress.progress(current / total, text=f"Analysing K={k_min + current - 1}… ({current}/{total})")

    analysis = find_optimal_k(embeddings, k_range=k_range, progress_callback=_progress_cb)
    progress.empty()

    st.session_state["persona_k_analysis"] = analysis
    st.toast(f"Best K by silhouette: **{analysis['best_k_silhouette']}**", icon="📊")


def _render_k_charts(analysis: dict):
    """Render the Elbow and Silhouette charts side by side."""
    k_values = analysis["k_values"]
    inertias = analysis["inertias"]
    silhouettes = analysis["silhouettes"]
    best_k = analysis["best_k_silhouette"]

    col_elbow, col_sil = st.columns(2)

    with col_elbow:
        st.markdown("##### Elbow Method (Inertia)")
        df_elbow = pd.DataFrame({"K": k_values, "Inertia": inertias}).set_index("K")
        st.line_chart(df_elbow, color="#6366f1")
        st.caption("Look for the *elbow* — where the curve bends and gains diminish.")

    with col_sil:
        st.markdown("##### Silhouette Score")
        df_sil = pd.DataFrame({"K": k_values, "Silhouette": silhouettes}).set_index("K")
        st.line_chart(df_sil, color="#22c55e")
        st.caption(f"Higher = better separation. **Peak at K = {best_k}** (score = {silhouettes[k_values.index(best_k)]:.4f})")

    st.info(f"🏆 **Recommended K = {best_k}** based on silhouette score. "
            f"You can still override this below.")


def _run_clustering(k: int):
    """Run KMeans with the chosen K."""
    from tools.persona_clustering import cluster_embeddings, cluster_stats

    embeddings = st.session_state["persona_embeddings"]

    with st.spinner(f"Clustering into {k} groups…"):
        labels = cluster_embeddings(embeddings, k=k)
        st.session_state["persona_labels"] = labels
        st.session_state["persona_k"] = k

        stats = cluster_stats(labels, k)
        st.session_state["persona_cluster_stats"] = stats

        # Initialize cluster names
        for c in range(k):
            if f"persona_cluster_name_{c}" not in st.session_state:
                st.session_state[f"persona_cluster_name_{c}"] = f"Cluster {c}"

        # Reset weights
        st.session_state["persona_weights"] = [round(1.0 / k, 3)] * k

    st.toast(f"Clustered into {k} groups", icon="✅")


def _save_persona(name: str, weights: list[float], k: int):
    """Append a persona definition to session_state."""
    if "saved_personas" not in st.session_state:
        st.session_state["saved_personas"] = []

    cluster_names = [
        st.session_state.get(f"persona_cluster_name_{i}", f"Cluster {i}")
        for i in range(k)
    ]

    st.session_state["saved_personas"].append({
        "name": name,
        "weights": list(weights),
        "cluster_names": cluster_names,
        "k": k,
    })


# ── Sub-clustering helpers ─────────────────────────────────────────────────────

def _run_sub_clustering(cluster_id: int, sub_k: int):
    """Run KMeans sub-clustering on a single top-level cluster."""
    from tools.persona_clustering import sub_cluster_embeddings, sub_cluster_stats

    embeddings = st.session_state["persona_embeddings"]
    labels = st.session_state["persona_labels"]
    emotion_probs = st.session_state.get("persona_emotion_probs")

    with st.spinner(f"Sub-clustering cluster {cluster_id} into {sub_k} groups…"):
        sub_labels = sub_cluster_embeddings(
            embeddings, labels, cluster_id, sub_k=sub_k,
        )
        st.session_state[f"persona_sub_labels_{cluster_id}"] = sub_labels
        st.session_state[f"persona_sub_k_{cluster_id}"] = sub_k

        sub_stats = sub_cluster_stats(sub_labels, sub_k, emotion_probs)
        st.session_state[f"persona_sub_stats_{cluster_id}"] = sub_stats

    st.toast(f"Sub-clustered into {sub_k} groups", icon="🔬")


def _render_sub_clusters(cluster_id: int):
    """Render sub-cluster stats, samples, LLM generation, and Save as Identity."""
    from tools.persona_clustering import (
        sub_cluster_sample, dominant_emotion,
        generate_personality_description,
        save_discovered_identity, load_discovered_identities,
        delete_discovered_identity,
    )

    sub_labels = st.session_state[f"persona_sub_labels_{cluster_id}"]
    sub_k = st.session_state[f"persona_sub_k_{cluster_id}"]
    sub_stats = st.session_state[f"persona_sub_stats_{cluster_id}"]
    emotion_probs = st.session_state.get("persona_emotion_probs")
    vad_vectors = st.session_state.get("persona_vad_vectors")
    has_emotions = bool(emotion_probs)
    cluster_name = st.session_state.get(
        f"persona_cluster_name_{cluster_id}", f"Cluster {cluster_id}"
    )

    # Sub-cluster stats table
    st.markdown(f"##### Sub-clusters of *{cluster_name}*")

    emoji_map = {
        "joy": "😊", "anger": "😠", "fear": "😨",
        "sadness": "😢", "neutral": "😐",
    }

    if has_emotions:
        cols_header = st.columns([1, 2, 2, 3])
        cols_header[0].markdown("**#**")
        cols_header[1].markdown("**Messages**")
        cols_header[2].markdown("**%**")
        cols_header[3].markdown("**Dominant Emotion**")
    else:
        cols_header = st.columns([1, 2, 2, 2, 2, 2])
        cols_header[0].markdown("**#**")
        cols_header[1].markdown("**Messages**")
        cols_header[2].markdown("**%**")
        cols_header[3].markdown("**V̄**")
        cols_header[4].markdown("**Ā**")
        cols_header[5].markdown("**D̄**")

    for sc_stat in sub_stats:
        sc = sc_stat["sub_cluster"]
        if has_emotions:
            cols = st.columns([1, 2, 2, 3])
            cols[0].write(f"Sub {sc}")
            cols[1].write(sc_stat["count"])
            cols[2].write(f"{sc_stat['pct']}%")
            if "dominant_emotion" in sc_stat:
                dom = sc_stat["dominant_emotion"]
                cols[3].write(f"{emoji_map.get(dom, '❓')} {dom}")
            else:
                cols[3].write("—")
        else:
            cols = st.columns([1, 2, 2, 2, 2, 2])
            cols[0].write(f"Sub {sc}")
            cols[1].write(sc_stat["count"])
            cols[2].write(f"{sc_stat['pct']}%")
            # Compute per-sub-cluster VAD means
            if vad_vectors is not None:
                sc_mask = sub_labels == sc
                sc_vad = vad_vectors[sc_mask]
                if len(sc_vad) > 0:
                    cols[3].write(f"{np.mean(sc_vad[:, 0]):.3f}")
                    cols[4].write(f"{np.mean(sc_vad[:, 1]):.3f}")
                    cols[5].write(f"{np.mean(sc_vad[:, 2]):.3f}")
                else:
                    cols[3].write("—")
                    cols[4].write("—")
                    cols[5].write("—")

    # Per-sub-cluster expanders with samples + LLM generation
    for sc_stat in sub_stats:
        sc = sc_stat["sub_cluster"]

        if has_emotions:
            dom_label = sc_stat.get("dominant_emotion", "unknown")
            dom_emoji = emoji_map.get(dom_label, "❓")
            expander_title = (
                f"Sub-cluster {sc} — {dom_emoji} {dom_label} "
                f"({sc_stat['count']} msgs, {sc_stat['pct']}%)"
            )
        else:
            # Direct VAD mode — show mean VAD in expander title
            vad_info = ""
            if vad_vectors is not None:
                sc_mask = sub_labels == sc
                sc_vad = vad_vectors[sc_mask]
                if len(sc_vad) > 0:
                    v, a, d = np.mean(sc_vad[:, 0]), np.mean(sc_vad[:, 1]), np.mean(sc_vad[:, 2])
                    vad_info = f"V={v:.2f} A={a:.2f} D={d:.2f}"
            expander_title = (
                f"Sub-cluster {sc} — {vad_info} "
                f"({sc_stat['count']} msgs, {sc_stat['pct']}%)"
            )

        with st.expander(expander_title, expanded=False):
            # Sample messages
            samples = sub_cluster_sample(
                st.session_state["persona_messages"],
                sub_labels,
                sc,
                n=20,
            )

            st.markdown(f"**Sample messages** (up to 20):")
            # Build all message cards, then wrap in a scrollable container
            _msg_cards = []
            for i, msg in enumerate(samples):
                date_str = msg.get("date", "")[:10]
                text = msg.get("text", "").replace("<", "<").replace(">", ">")
                badge = ""
                if has_emotions and msg.get("index") is not None:
                    idx = msg["index"]
                    if idx < len(emotion_probs):
                        dom = dominant_emotion(emotion_probs[idx])
                        badge = f" {emoji_map.get(dom, '❓')}{dom}"
                elif not has_emotions and msg.get("index") is not None:
                    idx = msg["index"]
                    if vad_vectors is not None and idx < len(vad_vectors):
                        v, a, d = vad_vectors[idx]
                        badge = f" V={v:.2f} A={a:.2f} D={d:.2f}"
                _msg_cards.append(
                    f"<div style='padding:6px 10px;margin:3px 0;"
                    f"border:1px solid rgba(128,128,128,0.25);"
                    f"border-radius:6px;font-size:0.85em'>"
                    f"<b>#{i+1}</b> <span style=\"color:gray\">{date_str}</span>"
                    f"{badge} — {text[:200]}</div>"
                )
            st.markdown(
                "<div style='max-height:400px;overflow-y:auto;"
                "padding-right:6px'>"
                + "\n".join(_msg_cards)
                + "</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # LLM personality generation
            gen_key = f"_gen_personality_{cluster_id}_{sc}"
            desc_key = f"_personality_desc_{cluster_id}_{sc}"

            col_gen, col_status = st.columns([1, 2])
            with col_gen:
                if st.button(
                    "🤖 Generate Personality",
                    key=gen_key,
                    use_container_width=True,
                    help="Use LLM to generate a personality description from these messages.",
                ):
                    try:
                        with st.spinner("Generating personality description…"):
                            model = st.session_state.get("model", None)
                            host = st.session_state.get("ollama_host", None)
                            desc = generate_personality_description(
                                sample_messages=samples,
                                cluster_name=cluster_name,
                                sub_cluster_id=sc,
                                emotion_summary=sc_stat.get("avg_emotions"),
                                model=model,
                                ollama_host=host,
                            )
                            st.session_state[desc_key] = desc
                            st.rerun()
                    except Exception as e:
                        st.error(f"LLM generation failed: {e}")

            # Show editable description if generated
            if desc_key in st.session_state and st.session_state[desc_key]:
                edited_desc = st.text_area(
                    "Personality description (edit as needed):",
                    value=st.session_state[desc_key],
                    height=120,
                    key=f"_edit_desc_{cluster_id}_{sc}",
                )

                # Extract name from description (pattern: "You are 'Name' - ...")
                import re
                name_match = re.search(r"['\"]([^'\"]+)['\"]", edited_desc)
                default_name = name_match.group(1) if name_match else f"Sub-{cluster_name}-{sc}"

                col_name, col_save = st.columns([2, 1])
                with col_name:
                    identity_name = st.text_input(
                        "Identity name:",
                        value=default_name,
                        key=f"_identity_name_{cluster_id}_{sc}",
                    )
                with col_save:
                    st.write("")  # spacer
                    if st.button(
                        "💾 Save as Identity",
                        key=f"_save_identity_{cluster_id}_{sc}",
                        type="primary",
                        use_container_width=True,
                    ):
                        save_discovered_identity(
                            name=identity_name,
                            description=edited_desc,
                            source_cluster=cluster_name,
                            source_sub_cluster=sc,
                        )
                        # Also update session_state
                        if "discovered_identities" not in st.session_state:
                            st.session_state["discovered_identities"] = {}
                        st.session_state["discovered_identities"][identity_name] = edited_desc
                        st.toast(f"Identity **{identity_name}** saved!", icon="💾")

    # ── Discovered Identities section ──────────────────────────────────────
    _render_discovered_identities()


def _render_discovered_identities():
    """Show all discovered identities with delete buttons."""
    from tools.persona_clustering import (
        load_discovered_identities, delete_discovered_identity,
    )

    # Load from file if not in session_state
    if "discovered_identities" not in st.session_state:
        stored = load_discovered_identities()
        st.session_state["discovered_identities"] = {
            name: info["description"] for name, info in stored.items()
        }

    identities = st.session_state.get("discovered_identities", {})
    if not identities:
        return

    st.markdown("---")
    st.markdown("#### 🧠 Discovered Identities")
    st.caption(
        "These personality facets were discovered from your messages. "
        "They are available in **Settings → Inner Deliberation Committee**."
    )

    for name, desc in list(identities.items()):
        with st.expander(f"**{name}**"):
            st.markdown(desc)
            if st.button(
                "🗑️ Delete",
                key=f"_del_identity_{name}",
                type="secondary",
            ):
                delete_discovered_identity(name)
                del st.session_state["discovered_identities"][name]
                st.toast(f"Identity **{name}** deleted.", icon="🗑️")
                st.rerun()
