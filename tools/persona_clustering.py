"""
tools/persona_clustering.py — Persona discovery via message clustering.

Pipeline:
  1. Load facebook_messages.json, keep only the user's own messages (≥ min_tokens).
     Each message carries a ``"language"`` field (``"en"``/``"fr"``) set during
     the extraction step (``tools/extract_facebook.py``).
  2. Classify emotions using language-specific models:
     - EN: j-hartmann/emotion-english-distilroberta-base (7 emotions)
     - FR: astrosbd/french_emotion_camembert (6 emotions)
     Optionally uses a **conversation sliding window** (``window_size`` > 1)
     to give the classifier conversational context: preceding messages from
     the same conversation are concatenated with ``[SEP]``.
  3. Harmonize to 5 common labels (anger, fear, joy, sadness, neutral).
  4. Map emotion probabilities to 3D VAD (Valence-Arousal-Dominance) vectors.
  5. Run KMeans(k) clustering on the VAD vectors.
  6. Expose helpers for the Streamlit UI to explore clusters and define persona
     distributions (weights summing to 1.0).

Alternative: embed_messages() still available for semantic (topic) clustering.

State is cached in session_state so the expensive classification step runs only once.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Ensure project root importable when run standalone
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import EMBEDDING_MODEL, DATA_DIR


# ── helpers ────────────────────────────────────────────────────────────────────

# Noise patterns — system messages, placeholders, and non-content artifacts
# extracted from the Facebook HTML export.
_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^click for audio",
        r"^you sent an attachment",
        r"^this message was unsent",
        r"^this poll is no longer available",
        r"missed your (audio |video )?call",
        r"^you called\b",
        r"^the (video )?call ended",
        r"^you (pinned|unpinned) a message",
        r"^you created the group",
        r"^you (joined|left) the (call|group|video chat)",
        r"^you (added|removed|named|changed)",
        r"^you are now connected",
        r"^you can now message and call each other",
        r"^you'?re now friends with",
        # Third-person system messages (other participants' actions)
        r"named the group\b",
        r"changed the group photo",
        r"added .+ as a group admin",
        r"removed .+ as a group admin",
        r"changed the chat theme",
        r"set the emoji to",
        r"created the group",
        r"changed the group name",
        r"left the group\b",
        r"joined the group\b",
        r"added .+ to the group\b",
        r"removed .+ from the group\b",
        r"^you waved at",
        r"^you set the nickname",
        r"^you set your nickname",
        r"^you started a (call|video chat)",
        r"^IP Address:",
        # Messages containing URLs (shared links, not personal text)
        r"https?://",
        r"www\.",
    ]
]

# Reaction-emoji characters used by Facebook
_REACTION_EMOJI = re.compile(
    r"[😆❤👍😢😮😡🥰💕🤣😂🔥💯👏🙏😍🥺😭💀🤔😊😘🥲😅😏🤗🤩😤😳🫡🫶🫠🤝✨💜💙💚🧡💛🖤🤍💗💖]"
)


def _detect_self_name(messages: list[dict]) -> str:
    """Return the most-frequent sender_name (= the data owner)."""
    counts = Counter(m.get("sender_name", "") for m in messages)
    return counts.most_common(1)[0][0]


def _is_noise(text: str) -> bool:
    """Return True if *text* matches a known noise / system-message pattern."""
    for pat in _NOISE_PATTERNS:
        if pat.search(text):
            return True
    return False


def _clean_text(text: str) -> str:
    """
    Clean a message text by removing known artifacts.

    Handles:
    - Exact-half duplicates (Facebook reaction rendering artifact):
      ``"some text 😆Name some text 😆Name"`` → ``"some text"``
    - Trailing ``IP Address: ...`` metadata
    - Trailing ``Click for audio`` placeholder from voice messages
    """
    # Strip trailing IP address metadata
    text = re.sub(r"\s*IP Address:\s*[\d.]+\s*$", "", text).strip()

    # Strip trailing "Click for audio" (voice-message placeholder appended to real text)
    text = re.sub(r"\s*Click for audio\s*$", "", text, flags=re.IGNORECASE).strip()

    # Detect exact-half duplication (reaction artifact or copy-paste)
    half = len(text) // 2
    if half > 15:
        first = text[:half].strip()
        second = text[half:].strip()
        if first == second:
            # Take the first half and strip any trailing reaction emoji + name
            # Pattern: "actual message 😆SomeName" → "actual message"
            cleaned = _REACTION_EMOJI.split(first)[0].strip()
            if len(cleaned) >= 10:
                return cleaned
            # If stripping left too little, keep the first half as-is
            return first

    return text


def _is_emoji_only_reaction(text: str) -> bool:
    """
    Return True if text is just a reaction emoji followed by a person's name.
    E.g. ``"❤Pierre Tognet-Bruchet"`` or ``"😆Gabija Vasaityte"``.
    """
    return bool(re.match(r"^[^\w\s]{1,3}\s*[A-Z][\w\s-]+$", text))


def _word_count(text: str) -> int:
    """Count words using simple whitespace split."""
    return len(text.split())


def load_my_messages(
    json_path: str | Path | None = None,
    min_tokens: int = 3,
    self_name: str | None = None,
) -> tuple[list[dict], str]:
    """
    Load facebook_messages.json and return only the owner's messages.

    Parameters
    ----------
    min_tokens : int
        Minimum number of whitespace-separated words a message must have
        (applied both before and after cleaning).  Default 3 keeps short
        replies like ``"lol ok cool"`` while filtering single-word noise.

    Applies noise filtering:
    - Removes system messages (unsent, attachments, calls, polls, etc.)
    - Removes voice-message placeholders ("Click for audio")
    - Cleans reaction-duplicate artifacts (text repeated with emoji+name)
    - Removes emoji-only reactions ("❤PersonName")

    Returns
    -------
    (filtered_messages, detected_self_name)
        Each message dict has keys: date, sender_name, source, text, conversation,
        language.  The ``language`` field (``"en"``/``"fr"``) comes from the JSON
        produced by the extraction step; if missing it defaults to ``"en"``.
    """
    if json_path is None:
        json_path = DATA_DIR / "facebook" / "facebook_messages.json"
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Facebook messages not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        messages = json.load(f)

    if not self_name:
        self_name = _detect_self_name(messages)

    my_msgs = []
    for m in messages:
        if m.get("sender_name") != self_name:
            continue
        raw = m.get("text", "").strip()
        if _word_count(raw) < min_tokens:
            continue
        if _is_noise(raw):
            continue
        cleaned = _clean_text(raw)
        if _word_count(cleaned) < min_tokens:
            continue
        if _is_emoji_only_reaction(cleaned):
            continue
        # Store the cleaned text back
        my_msgs.append({**m, "text": cleaned})

    return my_msgs, self_name


def embed_messages(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int = 256,
    progress_callback=None,
) -> np.ndarray:
    """
    Embed a list of texts using SentenceTransformer.

    Parameters
    ----------
    texts : list[str]
    model_name : str, optional – defaults to config.EMBEDDING_MODEL
    batch_size : int – encode batch size
    progress_callback : callable(current, total) or None

    Returns
    -------
    np.ndarray of shape (len(texts), dim)
    """
    from sentence_transformers import SentenceTransformer
    import torch

    model_name = model_name or EMBEDDING_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    all_embeddings = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embs)
        if progress_callback:
            progress_callback(min(start + batch_size, total), total)

    return np.vstack(all_embeddings)


# ── Emotion classification & VAD mapping ──────────────────────────────────────

# Model identifiers for language-specific emotion classifiers
_EN_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
_FR_EMOTION_MODEL = "astrosbd/french_emotion_camembert"

# Direct VAD regression model (BERT fine-tuned on EmoBank for V/A/D regression)
_VAD_DIRECT_MODEL = "RobroKools/vad-bert"

# Common 5-label set after harmonization
EMOTION_LABELS = ("anger", "fear", "joy", "sadness", "neutral")

# VAD coordinates per emotion (Russell circumplex + Warriner norms)
_VAD_MAP: dict[str, tuple[float, float, float]] = {
    #              Valence  Arousal  Dominance
    "joy":       (0.85,    0.65,    0.70),
    "anger":     (0.15,    0.85,    0.75),
    "fear":      (0.10,    0.80,    0.20),
    "sadness":   (0.15,    0.25,    0.20),
    "neutral":   (0.50,    0.30,    0.50),
}


def _harmonize_english(raw_scores: list[dict]) -> dict[str, float]:
    """
    Harmonize the 7-label English model output to the common 5-label set.

    Folds: disgust → anger, surprise → joy.  Keeps neutral as-is.
    Renormalizes so probabilities sum to 1.0.
    """
    probs = {item["label"]: item["score"] for item in raw_scores}
    harmonized = {
        "anger":   probs.get("anger", 0.0) + probs.get("disgust", 0.0),
        "fear":    probs.get("fear", 0.0),
        "joy":     probs.get("joy", 0.0) + probs.get("surprise", 0.0),
        "sadness": probs.get("sadness", 0.0),
        "neutral": probs.get("neutral", 0.0),
    }
    total = sum(harmonized.values())
    if total > 0:
        harmonized = {k: v / total for k, v in harmonized.items()}
    return harmonized


def _harmonize_french(
    raw_scores: list[dict],
    neutral_threshold: float = 0.4,
) -> dict[str, float]:
    """
    Harmonize the 6-label French model output to the common 5-label set.

    Model ``astrosbd/french_emotion_camembert`` outputs:
    joy, sad, anger, fear, surprise, neutral.

    Folds: surprise → joy, sad → sadness.
    Neutral is native — no synthetic injection needed.
    Renormalizes so probabilities sum to 1.0.
    """
    probs = {item["label"]: item["score"] for item in raw_scores}

    harmonized = {
        "anger":   probs.get("anger", 0.0),
        "fear":    probs.get("fear", 0.0),
        "joy":     probs.get("joy", 0.0) + probs.get("surprise", 0.0),
        "sadness": probs.get("sad", 0.0) + probs.get("sadness", 0.0),
        "neutral": probs.get("neutral", 0.0),
    }

    # Renormalize
    total = sum(harmonized.values())
    if total > 0:
        harmonized = {k: v / total for k, v in harmonized.items()}
    return harmonized


def _build_windowed_texts(
    texts: list[str],
    messages: list[dict] | None,
    window_size: int,
) -> list[str]:
    """
    Build context-windowed texts by concatenating previous messages within
    the same conversation using ``[SEP]`` as separator.

    When *window_size* is 1 (or *messages* is ``None``), returns *texts*
    unchanged (no context).

    Parameters
    ----------
    texts : list[str]
        The flat list of message texts (one per message).
    messages : list[dict] | None
        The corresponding message dicts.  Each must have ``"conversation"``
        and ``"date"`` keys.  If ``None``, windowing is skipped.
    window_size : int
        Number of messages to include in the context window (including the
        current message).  E.g. ``window_size=3`` means the current message
        plus up to 2 preceding messages from the same conversation.

    Returns
    -------
    list[str]
        Context-enriched texts, same length as *texts*.
    """
    if window_size <= 1 or messages is None:
        return texts

    # Group message indices by conversation, preserving original order
    from collections import defaultdict
    conv_groups: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(messages):
        conv = m.get("conversation", "")
        conv_groups[conv].append(i)

    # Sort each conversation group by date
    for conv, indices in conv_groups.items():
        indices.sort(key=lambda i: messages[i].get("date", ""))

    # Build position-in-conversation lookup: idx → position within its conv
    idx_to_pos: dict[int, int] = {}
    idx_to_conv_indices: dict[int, list[int]] = {}
    for conv, indices in conv_groups.items():
        for pos, idx in enumerate(indices):
            idx_to_pos[idx] = pos
            idx_to_conv_indices[idx] = indices

    # Build windowed texts
    windowed: list[str] = []
    for i in range(len(texts)):
        if i not in idx_to_pos:
            # Shouldn't happen, but fallback to no context
            windowed.append(texts[i])
            continue

        pos = idx_to_pos[i]
        conv_indices = idx_to_conv_indices[i]

        # Window: from max(0, pos - window_size + 1) to pos (inclusive)
        start_pos = max(0, pos - window_size + 1)
        window_indices = conv_indices[start_pos : pos + 1]

        # Concatenate with [SEP]
        window_texts = [texts[idx] for idx in window_indices]
        windowed.append(" [SEP] ".join(window_texts))

    return windowed


def classify_emotions(
    texts: list[str],
    languages: list[str],
    batch_size: int = 64,
    neutral_threshold: float = 0.4,
    progress_callback=None,
    messages: list[dict] | None = None,
    window_size: int = 1,
) -> list[dict[str, float]]:
    """
    Classify emotions using language-specific models.

    Pipeline:
      1. Use pre-detected *languages* (from ``detect_conversation_languages``).
      2. Optionally build conversation-aware sliding window context.
      3. Batch English messages → distilroberta (7 emotions → 5 harmonized).
      4. Batch French messages → distilcamembert (6 emotions + neutral → 5).
      5. Merge results in original order.

    Parameters
    ----------
    texts : list[str]
    languages : list[str]
        Pre-detected languages (``'en'``/``'fr'``), one per text.
        Use ``detect_conversation_languages()`` to obtain these efficiently.
    batch_size : int
    neutral_threshold : float
        For French model: if max emotion prob < this, inject neutral.
    progress_callback : callable(current, total) or None
    messages : list[dict] | None
        Original message dicts with ``"conversation"`` and ``"date"`` keys.
        Required when *window_size* > 1 to group and order messages within
        conversations.
    window_size : int
        Number of messages to include in the sliding context window
        (including the current message).  Default 1 = no context (original
        behaviour).  E.g. 3 = current message + up to 2 preceding messages
        from the same conversation, joined with ``[SEP]``.

    Returns
    -------
    list[dict[str, float]]
        Each dict has keys: anger, fear, joy, sadness, neutral.
    """
    from transformers import pipeline as hf_pipeline
    import torch

    # Build context-windowed texts if requested
    input_texts = _build_windowed_texts(texts, messages, window_size)

    # Debug: log windowing stats and sample texts
    if window_size > 1:
        n_windowed = sum(1 for orig, win in zip(texts, input_texts) if orig != win)
        print(f"[classify_emotions] window_size={window_size}, "
              f"{n_windowed}/{len(texts)} messages have context")
        # Print first 5 windowed examples
        _sample_count = 0
        for i, (orig, win) in enumerate(zip(texts, input_texts)):
            if orig != win and _sample_count < 5:
                print(f"  [sample {i}] original: {orig[:80]!r}")
                print(f"  [sample {i}] windowed: {win[:200]!r}")
                _sample_count += 1
    else:
        print(f"[classify_emotions] window_size=1 (no context), "
              f"{len(texts)} messages")

    # Split indices by language
    en_indices = [i for i, lang in enumerate(languages) if lang == "en"]
    fr_indices = [i for i, lang in enumerate(languages) if lang == "fr"]

    device = 0 if torch.cuda.is_available() else -1
    results: list[dict[str, float] | None] = [None] * len(texts)
    processed = 0
    total = len(texts)

    # ── English batch ──────────────────────────────────────────────────────
    if en_indices:
        en_pipe = hf_pipeline(
            "text-classification",
            model=_EN_EMOTION_MODEL,
            top_k=None,
            truncation=True,
            device=device,
        )
        en_texts = [input_texts[i] for i in en_indices]
        print(f"[classify_emotions] EN: {len(en_texts)} texts to classify")
        if en_texts:
            print(f"  [EN sample 0] {en_texts[0][:200]!r}")
        _en_logged = 0
        for batch_start in range(0, len(en_texts), batch_size):
            batch = en_texts[batch_start : batch_start + batch_size]
            batch_results = en_pipe(batch)
            for j, raw in enumerate(batch_results):
                idx = en_indices[batch_start + j]
                harmonized = _harmonize_english(raw)
                results[idx] = harmonized
                if _en_logged < 3:
                    # Compute VAD for this single message to show alongside
                    _vad_vec = sum(
                        harmonized.get(e, 0.0) * np.array(_VAD_MAP[e])
                        for e in EMOTION_LABELS
                    )
                    print(f"  [EN {idx}] input: {en_texts[batch_start + j][:150]!r}")
                    print(f"  [EN {idx}] → V={_vad_vec[0]:.3f}  A={_vad_vec[1]:.3f}  D={_vad_vec[2]:.3f}  "
                          f"(dominant: {max(harmonized, key=harmonized.get)})")
                    _en_logged += 1
            processed += len(batch)
            if progress_callback:
                progress_callback(processed, total)
        # Free memory
        del en_pipe

    # ── French batch ───────────────────────────────────────────────────────
    if fr_indices:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # Load tokenizer with use_fast=False to avoid tiktoken/sentencepiece
        # parsing errors on transformers ≥ 5.x (convert_slow_tokenizer tries
        # to read the .bpe.model as a tiktoken vocab and fails).
        fr_tokenizer = AutoTokenizer.from_pretrained(
            _FR_EMOTION_MODEL, use_fast=False,
        )
        fr_model = AutoModelForSequenceClassification.from_pretrained(
            _FR_EMOTION_MODEL,
        )
        fr_pipe = hf_pipeline(
            "text-classification",
            model=fr_model,
            tokenizer=fr_tokenizer,
            top_k=None,
            truncation=True,
            device=device,
        )
        fr_texts = [input_texts[i] for i in fr_indices]
        print(f"[classify_emotions] FR: {len(fr_texts)} texts to classify")
        if fr_texts:
            print(f"  [FR sample 0] {fr_texts[0][:200]!r}")
        _fr_logged = 0
        for batch_start in range(0, len(fr_texts), batch_size):
            batch = fr_texts[batch_start : batch_start + batch_size]
            batch_results = fr_pipe(batch)
            for j, raw in enumerate(batch_results):
                idx = fr_indices[batch_start + j]
                harmonized = _harmonize_french(raw, neutral_threshold)
                results[idx] = harmonized
                if _fr_logged < 3:
                    _vad_vec = sum(
                        harmonized.get(e, 0.0) * np.array(_VAD_MAP[e])
                        for e in EMOTION_LABELS
                    )
                    print(f"  [FR {idx}] input: {fr_texts[batch_start + j][:150]!r}")
                    print(f"  [FR {idx}] → V={_vad_vec[0]:.3f}  A={_vad_vec[1]:.3f}  D={_vad_vec[2]:.3f}  "
                          f"(dominant: {max(harmonized, key=harmonized.get)})")
                    _fr_logged += 1
            processed += len(batch)
            if progress_callback:
                progress_callback(processed, total)
        del fr_pipe, fr_model, fr_tokenizer

    # Safety: fill any None slots (shouldn't happen)
    default = {e: 0.2 for e in EMOTION_LABELS}
    results = [r if r is not None else default for r in results]

    return results  # type: ignore[return-value]


def emotions_to_vad(emotion_probs: list[dict[str, float]]) -> np.ndarray:
    """
    Map emotion probability dicts to 3D VAD (Valence-Arousal-Dominance) vectors.

    Each VAD vector is a weighted sum: ``VAD = Σ(p_i × VAD_i)``
    where probabilities come from the harmonized 5-label emotion output.

    Returns
    -------
    np.ndarray of shape ``(n, 3)`` with columns [Valence, Arousal, Dominance].
    """
    vad_matrix = np.array([_VAD_MAP[e] for e in EMOTION_LABELS])  # (5, 3)
    prob_matrix = np.array(
        [[probs.get(e, 0.0) for e in EMOTION_LABELS] for probs in emotion_probs]
    )  # (n, 5)
    result = prob_matrix @ vad_matrix  # (n, 3)

    # Debug: print VAD summary stats
    print(f"[emotions_to_vad] {result.shape[0]} messages → VAD vectors")
    print(f"  Valence:   mean={result[:, 0].mean():.4f}  std={result[:, 0].std():.4f}  "
          f"range=[{result[:, 0].min():.4f}, {result[:, 0].max():.4f}]")
    print(f"  Arousal:   mean={result[:, 1].mean():.4f}  std={result[:, 1].std():.4f}  "
          f"range=[{result[:, 1].min():.4f}, {result[:, 1].max():.4f}]")
    print(f"  Dominance: mean={result[:, 2].mean():.4f}  std={result[:, 2].std():.4f}  "
          f"range=[{result[:, 2].min():.4f}, {result[:, 2].max():.4f}]")

    return result


# ── Direct VAD classification (RoBERTa-VAD) ────────────────────────────────────

def classify_vad_direct(
    texts: list[str],
    messages: list[dict] | None = None,
    window_size: int = 1,
    batch_size: int = 32,
    progress_callback=None,
) -> np.ndarray:
    """
    Classify messages directly into VAD (Valence-Arousal-Dominance) space
    using the ``RobroKools/vad-bert`` regression model (BERT fine-tuned on
    EmoBank).

    Unlike the emotion-based pipeline, this model outputs continuous V, A, D
    scores directly — no intermediate emotion labels.  The model's
    ``problem_type`` is ``"regression"`` so the 3 logits **are** the V, A, D
    values (no softmax).

    Parameters
    ----------
    texts : list[str]
        Message texts to classify.
    messages : list[dict] | None
        Original message dicts with ``"conversation"`` and ``"date"`` keys.
        Required when *window_size* > 1.
    window_size : int
        Sliding context window size (1 = no context).
    batch_size : int
        Batch size for inference.
    progress_callback : callable(current, total) or None

    Returns
    -------
    np.ndarray of shape ``(n, 3)`` with columns [Valence, Arousal, Dominance].
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    # Build context-windowed texts if requested
    input_texts = _build_windowed_texts(texts, messages, window_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(input_texts)

    # Load the VAD regression model directly (not via pipeline, because
    # the text-classification pipeline applies softmax which is wrong for
    # a regression model).
    tokenizer = AutoTokenizer.from_pretrained(_VAD_DIRECT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(_VAD_DIRECT_MODEL)
    model.to(device)
    model.eval()

    print(f"[classify_vad_direct] model={_VAD_DIRECT_MODEL}, {total} texts, "
          f"window_size={window_size}, batch_size={batch_size}, device={device}")

    vad_results = np.zeros((total, 3), dtype=np.float32)
    processed = 0
    _logged = 0

    for batch_start in range(0, total, batch_size):
        batch = input_texts[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            # logits shape: (batch, 3) — raw V, A, D regression scores
            logits = outputs.logits.cpu().numpy()

        for j in range(len(batch)):
            idx = batch_start + j
            vad_results[idx, 0] = logits[j, 0]  # Valence
            vad_results[idx, 1] = logits[j, 1]  # Arousal
            vad_results[idx, 2] = logits[j, 2]  # Dominance

            if _logged < 3:
                print(f"  [VAD {idx}] input: {input_texts[idx][:150]!r}")
                print(f"  [VAD {idx}] → V={vad_results[idx, 0]:.3f}  "
                      f"A={vad_results[idx, 1]:.3f}  D={vad_results[idx, 2]:.3f}")
                _logged += 1

        processed += len(batch)
        if progress_callback:
            progress_callback(processed, total)

    del model, tokenizer

    # Summary stats
    print(f"[classify_vad_direct] Done — {total} messages → VAD vectors")
    print(f"  Valence:   mean={vad_results[:, 0].mean():.4f}  "
          f"std={vad_results[:, 0].std():.4f}  "
          f"range=[{vad_results[:, 0].min():.4f}, {vad_results[:, 0].max():.4f}]")
    print(f"  Arousal:   mean={vad_results[:, 1].mean():.4f}  "
          f"std={vad_results[:, 1].std():.4f}  "
          f"range=[{vad_results[:, 1].min():.4f}, {vad_results[:, 1].max():.4f}]")
    print(f"  Dominance: mean={vad_results[:, 2].mean():.4f}  "
          f"std={vad_results[:, 2].std():.4f}  "
          f"range=[{vad_results[:, 2].min():.4f}, {vad_results[:, 2].max():.4f}]")

    return vad_results


# ── LLM-based VAD classification ──────────────────────────────────────────────

def load_conversation_messages(
    json_path: str | Path | None = None,
    min_tokens: int = 3,
    selected_conversations: list[str] | None = None,
) -> tuple[list[dict], str, list[str]]:
    """
    Load facebook_messages.json and return **all** messages (all speakers)
    from selected conversations, sorted by date within each conversation.

    Unlike :func:`load_my_messages` which keeps only the user's messages,
    this function preserves the full dialogue so the LLM can see context
    from all participants.

    Parameters
    ----------
    json_path : path to the JSON file (defaults to DATA_DIR/facebook/facebook_messages.json)
    min_tokens : minimum word count for the user's target messages
    selected_conversations : list of conversation names to include.
        If ``None``, all conversations with ≥2 user messages are included.

    Returns
    -------
    (all_messages, self_name, available_conversations)
        - *all_messages*: list of message dicts (all speakers) from selected
          conversations, sorted by date within each conversation.
          Each dict has keys: date, sender_name, source, text, conversation, language.
        - *self_name*: detected owner name.
        - *available_conversations*: list of conversation names that have ≥2
          user messages (for the UI conversation selector).
    """
    if json_path is None:
        json_path = DATA_DIR / "facebook" / "facebook_messages.json"
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Facebook messages not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        messages = json.load(f)

    self_name = _detect_self_name(messages)

    # Group messages by conversation
    from collections import defaultdict
    conv_msgs: dict[str, list[dict]] = defaultdict(list)
    for m in messages:
        conv = m.get("conversation", "")
        conv_msgs[conv].append(m)

    # Sort each conversation by date
    for conv in conv_msgs:
        conv_msgs[conv].sort(key=lambda m: m.get("date", ""))

    # Find conversations with ≥2 user messages (after noise filtering)
    available_conversations: list[str] = []
    for conv, msgs in conv_msgs.items():
        user_count = 0
        for m in msgs:
            if m.get("sender_name") != self_name:
                continue
            raw = m.get("text", "").strip()
            if _word_count(raw) < min_tokens:
                continue
            if _is_noise(raw):
                continue
            user_count += 1
            if user_count >= 2:
                available_conversations.append(conv)
                break

    available_conversations.sort()

    # Filter to selected conversations
    if selected_conversations is not None:
        keep = set(selected_conversations)
    else:
        keep = set(available_conversations)

    # Build flat list of all messages from selected conversations
    all_msgs: list[dict] = []
    for conv in sorted(keep):
        if conv in conv_msgs:
            all_msgs.extend(conv_msgs[conv])

    print(f"[load_conversation_messages] {len(all_msgs)} messages from "
          f"{len(keep)} conversations, self_name={self_name!r}, "
          f"{len(available_conversations)} available conversations")

    return all_msgs, self_name, available_conversations


def classify_vad_llm(
    all_messages: list[dict],
    self_name: str,
    window_size: int = 5,
    min_tokens: int = 3,
    model: str = "llama3:8b",
    ollama_host: str | None = None,
    progress_callback=None,
    smooth_window: int = 0,
) -> tuple[list[dict], np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Classify **every** message in the conversation into VAD space using an LLM.

    Unlike the previous version that only classified the user's messages,
    this classifies ALL messages from ALL speakers (skipping only the very
    first message in each conversation, which has no preceding context).

    Parameters
    ----------
    all_messages : list[dict]
        All messages (all speakers) from selected conversations, sorted by
        date within each conversation.  Output of :func:`load_conversation_messages`.
    self_name : str
        The user's sender name (stored in results for speaker identification).
    window_size : int
        Number of preceding messages to include as context (not counting the
        target message itself).
    min_tokens : int
        Minimum word count for a message to be classified.  Shorter messages
        are still included in the context window but not classified themselves.
    model : str
        Ollama model name (e.g. ``"llama3:8b"``, ``"qwen2.5:7b"``).
    ollama_host : str | None
        Ollama host URL.  Defaults to ``config.DEFAULT_OLLAMA``.
    progress_callback : callable(current, total) or None
    smooth_window : int
        Rolling-mean smoothing window size (0 = no smoothing, 2–3 recommended).
        Applied per-conversation after classification.

    Returns
    -------
    (classified_messages, vad_vectors, vad_deltas, vad_smoothed)
        - *classified_messages*: list of message dicts that were classified,
          in chronological order within each conversation.  Each dict has an
          extra ``"speaker"`` key for convenience.
        - *vad_vectors*: np.ndarray of shape ``(n, 3)`` — raw VAD values.
        - *vad_deltas*: np.ndarray of shape ``(n, 3)`` — delta between
          consecutive points (first message in each conversation has delta=0).
        - *vad_smoothed*: np.ndarray of shape ``(n, 3)`` or ``None`` —
          rolling-mean smoothed VAD (``None`` if ``smooth_window < 2``).
    """
    import ollama as _ollama
    from config import DEFAULT_OLLAMA

    ollama_host = ollama_host or DEFAULT_OLLAMA
    client = _ollama.Client(host=ollama_host)

    # Group messages by conversation, preserving order
    from collections import defaultdict
    conv_msgs: dict[str, list[dict]] = defaultdict(list)
    for m in all_messages:
        conv_msgs[m.get("conversation", "")].append(m)

    # Build list of (target_message, context_window, conversation_name) tuples
    # Classify ALL messages (all speakers), skip only the first in each conv
    targets: list[tuple[dict, list[dict], str]] = []
    for conv, msgs in conv_msgs.items():
        if len(msgs) < 2:
            continue  # skip single-message conversations

        for pos in range(1, len(msgs)):
            raw = msgs[pos].get("text", "").strip()
            # Skip very short / noise messages (but keep them in context)
            if _word_count(raw) < min_tokens or _is_noise(raw):
                continue

            start = max(0, pos - window_size)
            context = msgs[start:pos]
            targets.append((msgs[pos], context, conv))

    total = len(targets)
    print(f"[classify_vad_llm] {total} messages (all speakers) to classify, "
          f"model={model}, window_size={window_size}")

    if total == 0:
        return [], np.zeros((0, 3), dtype=np.float32), None, None

    vad_results = np.zeros((total, 3), dtype=np.float32)
    classified_messages: list[dict] = []
    conv_boundaries: list[int] = []  # indices where a new conversation starts
    _logged = 0
    _current_conv = None

    for i, (target, context, conv) in enumerate(targets):
        # Track conversation boundaries for delta computation
        if conv != _current_conv:
            conv_boundaries.append(i)
            _current_conv = conv

        # Build conversation history string
        history_lines = []
        for m in context:
            sender = m.get("sender_name", "Unknown")
            text = m.get("text", "").strip()[:300]
            history_lines.append(f"[{sender}]: {text}")

        target_text = target.get("text", "").strip()[:300]
        target_sender = target.get("sender_name", "Unknown")
        history_lines.append(f"[{target_sender}]: {target_text}")

        conversation_block = "\n".join(history_lines)

        prompt = (
            "Given the conversation:\n"
            f"{conversation_block}\n\n"
            "What is the emotional tone of the last speaker turn?\n"
            "Return ONLY a JSON object with these exact keys and ranges:\n"
            '{"valence": <float from -1.0 to 1.0>, '
            '"arousal": <float from 0.0 to 1.0>, '
            '"dominance": <float from 0.0 to 1.0>}\n'
            "Do not include any other text, explanation, or markdown formatting."
        )

        vad = _call_llm_for_vad(client, model, prompt)

        # Retry once on failure with stricter prompt
        if vad is None:
            retry_prompt = (
                "You must respond with ONLY a JSON object. No other text.\n"
                "Example: {\"valence\": 0.3, \"arousal\": 0.5, \"dominance\": 0.6}\n\n"
                f"Conversation:\n{conversation_block}\n\n"
                "Emotional tone of the last turn as JSON:"
            )
            vad = _call_llm_for_vad(client, model, retry_prompt)

        if vad is None:
            vad = (0.0, 0.5, 0.5)
            if _logged < 3:
                print(f"  [LLM-VAD {i}] FALLBACK — could not parse LLM response")

        vad_results[i, 0] = vad[0]
        vad_results[i, 1] = vad[1]
        vad_results[i, 2] = vad[2]

        # Store message with speaker info
        cleaned = _clean_text(target_text)
        classified_messages.append({
            **target,
            "text": cleaned,
            "speaker": target_sender,
            "conversation": conv,
        })

        if _logged < 5:
            print(f"  [LLM-VAD {i}] [{target_sender}] context={len(context)} msgs, "
                  f"target: {target_text[:80]!r}")
            print(f"  [LLM-VAD {i}] → V={vad[0]:.3f}  A={vad[1]:.3f}  D={vad[2]:.3f}")
            _logged += 1

        if progress_callback:
            progress_callback(i + 1, total)

    # ── Compute deltas between consecutive points ──────────────────────────
    vad_deltas = np.zeros_like(vad_results)
    conv_boundary_set = set(conv_boundaries)
    for i in range(1, total):
        if i not in conv_boundary_set:
            vad_deltas[i] = vad_results[i] - vad_results[i - 1]
        # else: delta stays 0 (first message in a new conversation)

    # ── Rolling mean smoothing ─────────────────────────────────────────────
    vad_smoothed: np.ndarray | None = None
    if smooth_window >= 2:
        vad_smoothed = np.zeros_like(vad_results)
        # Smooth per-conversation to avoid cross-conversation bleed
        # Build conversation ranges
        conv_ranges: list[tuple[int, int]] = []
        for ci, start_idx in enumerate(conv_boundaries):
            end_idx = conv_boundaries[ci + 1] if ci + 1 < len(conv_boundaries) else total
            conv_ranges.append((start_idx, end_idx))

        for start_idx, end_idx in conv_ranges:
            segment = vad_results[start_idx:end_idx]
            n_seg = len(segment)
            if n_seg <= 1:
                vad_smoothed[start_idx:end_idx] = segment
                continue
            # Simple rolling mean (centered, with edge padding)
            for dim in range(3):
                vals = segment[:, dim]
                smoothed = np.convolve(
                    vals,
                    np.ones(smooth_window) / smooth_window,
                    mode="same",
                )
                vad_smoothed[start_idx:end_idx, dim] = smoothed

    # Summary stats
    print(f"[classify_vad_llm] Done — {total} messages (all speakers) → VAD vectors")
    for dim, name in enumerate(["Valence", "Arousal", "Dominance"]):
        print(f"  {name:10s}: mean={vad_results[:, dim].mean():.4f}  "
              f"std={vad_results[:, dim].std():.4f}  "
              f"range=[{vad_results[:, dim].min():.4f}, "
              f"{vad_results[:, dim].max():.4f}]")
    print(f"  Deltas — mean abs: V={np.abs(vad_deltas[:, 0]).mean():.4f}  "
          f"A={np.abs(vad_deltas[:, 1]).mean():.4f}  "
          f"D={np.abs(vad_deltas[:, 2]).mean():.4f}")
    if vad_smoothed is not None:
        print(f"  Smoothing applied: window={smooth_window}")

    return classified_messages, vad_results, vad_deltas, vad_smoothed


def _call_llm_for_vad(
    client,
    model: str,
    prompt: str,
) -> tuple[float, float, float] | None:
    """
    Call Ollama and parse the JSON response for V/A/D values.

    Returns ``(valence, arousal, dominance)`` or ``None`` on failure.
    """
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        raw = resp["message"]["content"].strip()

        # Extract JSON from response (LLMs sometimes wrap in markdown)
        match = re.search(r"\{[^}]+\}", raw)
        if not match:
            return None

        data = json.loads(match.group())

        v = float(data.get("valence", 0.0))
        a = float(data.get("arousal", 0.5))
        d = float(data.get("dominance", 0.5))

        # Validate ranges
        v = max(-1.0, min(1.0, v))
        a = max(0.0, min(1.0, a))
        d = max(0.0, min(1.0, d))

        return (v, a, d)

    except Exception as e:
        print(f"  [_call_llm_for_vad] Error: {e}")
        return None


def dominant_emotion(probs: dict[str, float]) -> str:
    """Return the emotion label with the highest probability."""
    return max(probs, key=probs.get)  # type: ignore[arg-type]


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: range | None = None,
    random_state: int = 42,
    progress_callback=None,
) -> dict:
    """
    Run the Elbow method (inertia) and Silhouette score for a range of K values.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n, dim)
    k_range : range, optional – defaults to range(2, 20)
    random_state : int
    progress_callback : callable(current, total) or None

    Returns
    -------
    dict with keys:
        k_values : list[int]
        inertias : list[float]       — for the Elbow plot
        silhouettes : list[float]    — for the Silhouette plot
        best_k_silhouette : int      — K with the highest silhouette score
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if k_range is None:
        k_range = range(2, 20)

    k_values = list(k_range)
    inertias: list[float] = []
    silhouettes: list[float] = []
    total = len(k_values)

    for i, k in enumerate(k_values):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(embeddings, labels, sample_size=min(10_000, len(embeddings))))
        silhouettes.append(sil)
        if progress_callback:
            progress_callback(i + 1, total)

    best_idx = int(np.argmax(silhouettes))
    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k_silhouette": k_values[best_idx],
    }


def cluster_embeddings(
    embeddings: np.ndarray,
    k: int = 7,
    random_state: int = 42,
) -> np.ndarray:
    """
    Run KMeans on the embedding matrix.

    Returns
    -------
    labels : np.ndarray of shape (n,) with cluster ids 0..k-1
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels


def cluster_sample(
    messages: list[dict],
    labels: np.ndarray,
    cluster_id: int,
    n: int = 50,
    random_state: int = 42,
) -> list[dict]:
    """
    Return up to *n* random messages from *cluster_id*.
    Each returned dict has the original message keys plus ``cluster``.
    """
    rng = np.random.RandomState(random_state)
    indices = np.where(labels == cluster_id)[0]
    if len(indices) > n:
        indices = rng.choice(indices, size=n, replace=False)
    indices = sorted(indices)
    return [
        {**messages[i], "cluster": int(cluster_id), "index": int(i)}
        for i in indices
    ]


def cluster_stats(labels: np.ndarray, k: int) -> list[dict]:
    """
    Return per-cluster statistics.

    Returns list of dicts with keys: cluster, count, pct.
    """
    total = len(labels)
    stats = []
    for c in range(k):
        count = int(np.sum(labels == c))
        stats.append({
            "cluster": c,
            "count": count,
            "pct": round(count / total * 100, 1) if total else 0,
        })
    return stats


def validate_distribution(weights: list[float], tol: float = 0.01) -> bool:
    """Check that weights sum to 1.0 (within tolerance)."""
    return abs(sum(weights) - 1.0) <= tol


# ── Sub-clustering ─────────────────────────────────────────────────────────────

def sub_cluster_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    sub_k: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """
    Run KMeans on the subset of embeddings belonging to *cluster_id*.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n, dim)
    labels : np.ndarray of shape (n,) — top-level cluster labels
    cluster_id : int — which top-level cluster to sub-divide
    sub_k : int — number of sub-clusters (default 3)
    random_state : int

    Returns
    -------
    sub_labels : np.ndarray of shape (n,)
        -1 for messages NOT in *cluster_id*,
        0..sub_k-1 for messages in *cluster_id*.
    """
    from sklearn.cluster import KMeans

    indices = np.where(labels == cluster_id)[0]
    if len(indices) < sub_k:
        # Not enough points — assign all to sub-cluster 0
        sub_labels = np.full(len(labels), -1, dtype=int)
        sub_labels[indices] = 0
        return sub_labels

    subset = embeddings[indices]
    km = KMeans(n_clusters=sub_k, random_state=random_state, n_init=10)
    sub_ids = km.fit_predict(subset)

    sub_labels = np.full(len(labels), -1, dtype=int)
    sub_labels[indices] = sub_ids
    return sub_labels


def sub_cluster_stats(
    sub_labels: np.ndarray,
    sub_k: int,
    emotion_probs: list[dict[str, float]] | None = None,
) -> list[dict]:
    """
    Per-sub-cluster statistics.

    Parameters
    ----------
    sub_labels : np.ndarray — output of sub_cluster_embeddings()
    sub_k : int — number of sub-clusters
    emotion_probs : optional list of emotion probability dicts (same length as sub_labels)

    Returns
    -------
    list of dicts with keys: sub_cluster, count, pct, and optionally
    dominant_emotion and avg_emotions.
    """
    # Only count messages that belong to the sub-clustered parent (not -1)
    member_mask = sub_labels >= 0
    total = int(np.sum(member_mask))
    stats = []

    for sc in range(sub_k):
        sc_mask = sub_labels == sc
        count = int(np.sum(sc_mask))
        entry: dict = {
            "sub_cluster": sc,
            "count": count,
            "pct": round(count / total * 100, 1) if total else 0,
        }

        if emotion_probs is not None:
            sc_indices = np.where(sc_mask)[0]
            if len(sc_indices) > 0:
                avg_emo: dict[str, float] = {}
                for idx in sc_indices:
                    for emo, prob in emotion_probs[idx].items():
                        avg_emo[emo] = avg_emo.get(emo, 0.0) + prob
                avg_emo = {e: round(v / len(sc_indices), 3) for e, v in avg_emo.items()}
                entry["avg_emotions"] = avg_emo
                entry["dominant_emotion"] = max(avg_emo, key=avg_emo.get)  # type: ignore[arg-type]

        stats.append(entry)

    return stats


def sub_cluster_sample(
    messages: list[dict],
    sub_labels: np.ndarray,
    sub_cluster_id: int,
    n: int = 30,
    random_state: int = 42,
) -> list[dict]:
    """
    Return up to *n* random messages from *sub_cluster_id*.

    Each returned dict has the original message keys plus ``sub_cluster``
    and ``index`` (position in the full messages list).
    """
    rng = np.random.RandomState(random_state)
    indices = np.where(sub_labels == sub_cluster_id)[0]
    if len(indices) > n:
        indices = rng.choice(indices, size=n, replace=False)
    indices = sorted(indices)
    return [
        {**messages[i], "sub_cluster": int(sub_cluster_id), "index": int(i)}
        for i in indices
    ]


# ── LLM personality generation ────────────────────────────────────────────────

def generate_personality_description(
    sample_messages: list[dict],
    cluster_name: str,
    sub_cluster_id: int,
    emotion_summary: dict[str, float] | None = None,
    model: str | None = None,
    ollama_host: str | None = None,
) -> str:
    """
    Send sample messages to Ollama and generate a personality description
    in the style of the existing IDENTITIES (IFS-style, second person).

    Parameters
    ----------
    sample_messages : list of message dicts (must have 'text' key)
    cluster_name : human-readable name of the parent cluster
    sub_cluster_id : which sub-cluster within the parent
    emotion_summary : optional dict of average emotion probabilities
    model : Ollama model name (defaults to config.DEFAULT_MODEL)
    ollama_host : Ollama host URL (defaults to config.DEFAULT_OLLAMA)

    Returns
    -------
    str — the generated personality description
    """
    import ollama as _ollama
    from config import DEFAULT_MODEL, DEFAULT_OLLAMA

    model = model or DEFAULT_MODEL
    ollama_host = ollama_host or DEFAULT_OLLAMA

    # Build message texts (limit to 25 for prompt size)
    texts = [m.get("text", "") for m in sample_messages[:25]]
    messages_block = "\n".join(f"- {t[:300]}" for t in texts if t.strip())

    emotion_ctx = ""
    if emotion_summary:
        top_emotions = sorted(emotion_summary.items(), key=lambda x: x[1], reverse=True)[:3]
        emotion_ctx = (
            f"\nEmotion profile: "
            + ", ".join(f"{e} ({p:.0%})" for e, p in top_emotions)
        )

    prompt = (
        "You are analyzing a person's messaging patterns to discover their inner personality facets. "
        "Below are sample messages from a specific behavioral cluster.\n\n"
        "Based on these messages, write a 2-3 sentence personality description in second person "
        "that captures the communication style, emotional tone, and behavioral patterns.\n\n"
        "The description MUST follow this exact format:\n"
        "\"You are '[Name]' - [description of this personality facet]. "
        "[More detail about how this part communicates and what drives it].\"\n\n"
        "The name should be a short, evocative title like 'The Empathetic Listener', "
        "'The Analytical Debater', 'The Playful Joker', etc.\n\n"
        f"Cluster context: {cluster_name}, sub-cluster {sub_cluster_id}{emotion_ctx}\n\n"
        f"Sample messages:\n{messages_block}\n\n"
        "Write ONLY the personality description, nothing else. No preamble, no explanation."
    )

    client = _ollama.Client(host=ollama_host)
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    return resp["message"]["content"].strip()


# ── Discovered identity persistence ───────────────────────────────────────────

_IDENTITIES_FILE = Path(".states/discovered_identities.json")


def load_discovered_identities() -> dict[str, dict]:
    """
    Load discovered identities from the JSON persistence file.

    Returns
    -------
    dict mapping identity name → {description, source_cluster,
    source_sub_cluster, created_at}
    """
    if not _IDENTITIES_FILE.exists():
        return {}
    try:
        with open(_IDENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_discovered_identity(
    name: str,
    description: str,
    source_cluster: str,
    source_sub_cluster: int,
) -> None:
    """
    Save (or update) a discovered identity to the JSON persistence file.

    Uses atomic write (write to temp, then rename) to avoid corruption.
    """
    from datetime import datetime, timezone

    identities = load_discovered_identities()
    identities[name] = {
        "description": description,
        "source_cluster": source_cluster,
        "source_sub_cluster": source_sub_cluster,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    _IDENTITIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _IDENTITIES_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(identities, f, indent=2, ensure_ascii=False)
    tmp.rename(_IDENTITIES_FILE)


def delete_discovered_identity(name: str) -> bool:
    """
    Remove a discovered identity from the JSON persistence file.

    Returns True if the identity was found and removed, False otherwise.
    """
    identities = load_discovered_identities()
    if name not in identities:
        return False

    del identities[name]

    _IDENTITIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _IDENTITIES_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(identities, f, indent=2, ensure_ascii=False)
    tmp.rename(_IDENTITIES_FILE)
    return True
