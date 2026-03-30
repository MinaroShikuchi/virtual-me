"""
ui/training_compare.py — Training run comparison: parse CLI commands & logs,
compare hyperparameters side-by-side, overlay training curves, and detect
optimal stopping points.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── Persistence ───────────────────────────────────────────────────────────────

_RUNS_FILE = Path(".states/training_runs.json")


def _load_runs() -> list[dict]:
    """Load all saved training runs from disk."""
    if not _RUNS_FILE.exists():
        return []
    try:
        with open(_RUNS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("runs", [])
    except (json.JSONDecodeError, OSError):
        return []


def _save_runs(runs: list[dict]) -> None:
    """Persist training runs to disk."""
    _RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_RUNS_FILE, "w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f, indent=2, ensure_ascii=False)


# ── CLI Command Parser ───────────────────────────────────────────────────────

# Maps CLI flags to (param_key, type_converter)
_CLI_FLAGS: dict[str, tuple[str, type]] = {
    "--base-model":     ("base_model",     str),
    "--data":           ("data",           str),
    "--output":         ("output",         str),
    "--epochs":         ("epochs",         int),
    "--batch-size":     ("batch_size",     int),
    "--lr":             ("lr",             float),
    "--lora-rank":      ("lora_rank",      int),
    "--lora-alpha":     ("lora_alpha",     int),
    "--lora-dropout":   ("lora_dropout",   float),
    "--max-seq-length": ("max_seq_length", int),
    "--grad-accum":     ("grad_accum",     int),
}

# Defaults matching tools/finetune_lora.py
_PARAM_DEFAULTS: dict[str, Any] = {
    "data":           "./data/facebook/finetune_data.jsonl",
    "output":         "./models/adapters/my-lora",
    "epochs":         3,
    "batch_size":     2,
    "lr":             2e-4,
    "lora_rank":      8,
    "lora_alpha":     16,
    "lora_dropout":   0.0,
    "max_seq_length": 512,
    "grad_accum":     4,
    "use_4bit":       True,
}


def parse_cli_command(command: str) -> dict[str, Any]:
    """Parse a finetune_lora.py CLI command string into a params dict."""
    params = dict(_PARAM_DEFAULTS)

    # Normalise: collapse newlines / backslash continuations
    command = command.replace("\\\n", " ").replace("\n", " ").strip()

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--no-4bit":
            params["use_4bit"] = False
            i += 1
            continue
        if tok == "--resume":
            params["resume"] = True
            i += 1
            continue
        if tok in _CLI_FLAGS and i + 1 < len(tokens):
            key, converter = _CLI_FLAGS[tok]
            try:
                params[key] = converter(tokens[i + 1])
            except (ValueError, TypeError):
                params[key] = tokens[i + 1]
            i += 2
            continue
        i += 1

    return params


# ── Training Log Parser ──────────────────────────────────────────────────────

# Matches Python dict-style log lines: {'loss': '0.7359', ...}
_LOG_LINE_RE = re.compile(r"\{[^}]*'loss'[^}]*\}")


def parse_training_logs(log_text: str, logging_steps: int = 10) -> list[dict]:
    """Parse training log output into a list of metric dicts.

    Handles both Python dict-style ``{'key': 'val'}`` and JSON
    ``{"key": "val"}`` formats.
    """
    entries: list[dict] = []

    for line in log_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Try to find a dict-like structure in the line
        match = _LOG_LINE_RE.search(line)
        if match:
            raw = match.group(0)
        elif line.startswith("{") and line.endswith("}"):
            raw = line
        else:
            continue

        # Convert Python dict syntax to JSON
        raw = raw.replace("'", '"')

        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue

        # Cast string values to floats where possible
        parsed: dict[str, Any] = {}
        for k, v in entry.items():
            if isinstance(v, str):
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            else:
                parsed[k] = v

        # Assign step number if not present
        if "step" not in parsed:
            parsed["step"] = (len(entries) + 1) * logging_steps

        entries.append(parsed)

    return entries


# ── Optimal Stopping Analysis ────────────────────────────────────────────────

def _moving_average(values: list[float], window: int = 5) -> list[float]:
    """Simple moving average with edge padding."""
    if len(values) <= window:
        return values[:]
    result = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


def analyse_stopping(log_entries: list[dict]) -> dict[str, Any]:
    """Analyse training logs to find optimal stopping points."""
    if not log_entries:
        return {}

    losses = [(e.get("epoch", e.get("step", i)), e["loss"])
              for i, e in enumerate(log_entries) if "loss" in e]
    accuracies = [(e.get("epoch", e.get("step", i)), e["mean_token_accuracy"])
                  for i, e in enumerate(log_entries) if "mean_token_accuracy" in e]

    result: dict[str, Any] = {}

    if losses:
        loss_vals = [v for _, v in losses]
        min_loss = min(loss_vals)
        min_idx = loss_vals.index(min_loss)
        result["best_loss"] = min_loss
        result["best_loss_epoch"] = losses[min_idx][0]
        result["final_loss"] = loss_vals[-1]

        # Overfitting detection: smoothed loss increases after minimum
        smoothed = _moving_average(loss_vals, window=5)
        if min_idx < len(smoothed) - 3:
            tail = smoothed[min_idx:]
            increasing_count = sum(1 for i in range(1, len(tail)) if tail[i] > tail[i - 1])
            result["overfitting"] = increasing_count > len(tail) * 0.6
            if result["overfitting"]:
                result["overfit_after_epoch"] = losses[min_idx][0]
        else:
            result["overfitting"] = False

    if accuracies:
        acc_vals = [v for _, v in accuracies]
        max_acc = max(acc_vals)
        max_idx = acc_vals.index(max_acc)
        result["peak_accuracy"] = max_acc
        result["peak_accuracy_epoch"] = accuracies[max_idx][0]
        result["final_accuracy"] = acc_vals[-1]

    return result


# ── Main Render Function ─────────────────────────────────────────────────────

def render_training_compare():
    """Render the full Training Comparison page."""
    st.markdown("### :material/compare_arrows: Training Run Comparison")
    st.caption(
        "Compare LoRA fine-tuning runs side-by-side. Paste the CLI command "
        "and training logs to create entries, then overlay training curves "
        "to find the best hyperparameters."
    )

    runs = _load_runs()

    # ── Section 1: Add New Run ────────────────────────────────────────────
    _render_add_run(runs)

    # Reload in case a run was just added
    runs = _load_runs()

    if not runs:
        st.info("No training runs saved yet. Add one above to get started.")
        return

    # ── Section 2: Runs Comparison Table ──────────────────────────────────
    selected_ids = _render_runs_table(runs)

    # Reload in case a run was deleted
    runs = _load_runs()
    if not runs:
        return

    # ── Section 3: Training Curves ────────────────────────────────────────
    selected_runs = [r for r in runs if r["id"] in selected_ids]
    if selected_runs:
        _render_training_curves(selected_runs)

        # ── Section 4: Optimal Stopping Analysis ─────────────────────────
        _render_stopping_analysis(selected_runs)


# ── Section 1: Add New Run ────────────────────────────────────────────────────

def _render_add_run(runs: list[dict]) -> None:
    with st.expander("➕ Add New Training Run", expanded=not runs, icon=":material/add_circle:"):
        col1, col2 = st.columns([2, 1])
        with col1:
            run_name = st.text_input(
                "Run name (optional)",
                placeholder="e.g. Ministral-3 r32 lr1e-4",
                key="tc_run_name",
                help="Leave empty to auto-generate from base model and params.",
            )
        with col2:
            logging_steps = st.number_input(
                "Logging steps",
                min_value=1, max_value=1000, value=10, step=1,
                key="tc_logging_steps",
                help="The --logging-steps value used during training (for step numbering).",
            )

        cli_command = st.text_area(
            "CLI command",
            height=80,
            placeholder="./tools/finetune_lora.py --base-model unsloth/Ministral-3-8B-Instruct-2512-bnb-4bit --epochs 2 --batch-size 1 ...",
            key="tc_cli_command",
            help="Paste the full command line used to launch training.",
        )

        log_text = st.text_area(
            "Training logs",
            height=200,
            placeholder="{'loss': '0.7359', 'grad_norm': '0.04636', 'learning_rate': '7.26e-05', ...}\n{'loss': '0.7337', ...}",
            key="tc_log_text",
            help="Paste the training log output. Each line should contain a dict with loss, accuracy, etc.",
        )

        notes = st.text_area(
            "Notes (optional)",
            height=60,
            placeholder="First attempt with higher rank...",
            key="tc_notes",
        )

        # Parse preview
        if cli_command.strip():
            params = parse_cli_command(cli_command)
            with st.container(border=True):
                st.markdown("**Parsed parameters:**")
                param_cols = st.columns(4)
                display_params = [
                    ("Base Model", params.get("base_model", "—")),
                    ("Epochs", params.get("epochs", "—")),
                    ("Batch Size", params.get("batch_size", "—")),
                    ("Grad Accum", params.get("grad_accum", "—")),
                    ("Eff. Batch", params.get("batch_size", 1) * params.get("grad_accum", 1)),
                    ("LR", params.get("lr", "—")),
                    ("LoRA Rank", params.get("lora_rank", "—")),
                    ("LoRA Alpha", params.get("lora_alpha", "—")),
                    ("Dropout", params.get("lora_dropout", "—")),
                    ("Max Seq Len", params.get("max_seq_length", "—")),
                    ("4-bit", "✅" if params.get("use_4bit") else "❌"),
                ]
                for i, (label, val) in enumerate(display_params):
                    param_cols[i % 4].markdown(f"**{label}:** `{val}`")

        if log_text.strip():
            log_entries = parse_training_logs(log_text, logging_steps=logging_steps)
            st.caption(f"Parsed **{len(log_entries)}** log entries")

        # Save button
        if st.button("💾 Save Run", key="tc_save_run", type="primary",
                     disabled=not cli_command.strip()):
            params = parse_cli_command(cli_command)
            log_entries = parse_training_logs(log_text, logging_steps=logging_steps) if log_text.strip() else []

            # Auto-generate name if empty
            if not run_name.strip():
                model_short = params.get("base_model", "unknown").split("/")[-1][:30]
                run_name = f"{model_short} r{params.get('lora_rank', '?')} lr{params.get('lr', '?')}"

            run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

            new_run = {
                "id": run_id,
                "name": run_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "command": cli_command.strip(),
                "params": params,
                "log_entries": log_entries,
                "notes": notes.strip(),
            }

            runs.append(new_run)
            _save_runs(runs)
            st.success(f"Saved run **{run_name}** with {len(log_entries)} log entries.")
            st.rerun()


# ── Section 2: Runs Comparison Table ──────────────────────────────────────────

def _render_runs_table(runs: list[dict]) -> set[str]:
    """Render the comparison table and return the set of selected run IDs."""
    st.markdown("---")
    st.markdown("### :material/table_chart: Runs Comparison")

    # Build comparison dataframe
    rows = []
    for run in runs:
        p = run.get("params", {})
        logs = run.get("log_entries", [])
        analysis = analyse_stopping(logs)

        rows.append({
            "id": run["id"],
            "Name": run.get("name", run["id"]),
            "Base Model": _short_model_name(p.get("base_model", "—")),
            "Epochs": p.get("epochs", "—"),
            "Batch": p.get("batch_size", "—"),
            "Grad Accum": p.get("grad_accum", "—"),
            "Eff. Batch": p.get("batch_size", 1) * p.get("grad_accum", 1),
            "LR": p.get("lr", "—"),
            "Rank": p.get("lora_rank", "—"),
            "Alpha": p.get("lora_alpha", "—"),
            "Dropout": p.get("lora_dropout", "—"),
            "Max Seq": p.get("max_seq_length", "—"),
            "4-bit": "✅" if p.get("use_4bit") else "❌",
            "Best Loss": f"{analysis['best_loss']:.4f}" if "best_loss" in analysis else "—",
            "Final Loss": f"{analysis['final_loss']:.4f}" if "final_loss" in analysis else "—",
            "Peak Acc": f"{analysis['peak_accuracy']:.2%}" if "peak_accuracy" in analysis else "—",
            "Final Acc": f"{analysis['final_accuracy']:.2%}" if "final_accuracy" in analysis else "—",
            "Log Steps": len(logs),
        })

    df = pd.DataFrame(rows)

    # Selection checkboxes
    st.caption("Select runs to compare in charts below:")
    selected_ids: set[str] = set()

    for i, run in enumerate(runs):
        col_sel, col_name, col_info, col_del = st.columns([0.5, 3, 5, 1])
        with col_sel:
            if st.checkbox("Select", value=True, key=f"tc_sel_{run['id']}", label_visibility="collapsed"):
                selected_ids.add(run["id"])
        with col_name:
            st.markdown(f"**{run.get('name', run['id'])}**")
            st.caption(run.get("notes", "")[:80] if run.get("notes") else "")
        with col_info:
            p = run.get("params", {})
            analysis = analyse_stopping(run.get("log_entries", []))
            info_parts = [
                f"r{p.get('lora_rank', '?')}",
                f"α{p.get('lora_alpha', '?')}",
                f"lr={p.get('lr', '?')}",
                f"bs={p.get('batch_size', '?')}×{p.get('grad_accum', '?')}",
                f"ep={p.get('epochs', '?')}",
            ]
            if "best_loss" in analysis:
                info_parts.append(f"best_loss={analysis['best_loss']:.4f}")
            if "peak_accuracy" in analysis:
                info_parts.append(f"peak_acc={analysis['peak_accuracy']:.2%}")
            st.caption(" · ".join(info_parts))
        with col_del:
            if st.button("🗑️", key=f"tc_del_{run['id']}", help=f"Delete run: {run.get('name', run['id'])}"):
                runs_updated = [r for r in runs if r["id"] != run["id"]]
                _save_runs(runs_updated)
                st.rerun()

    # Full parameter comparison table
    if len(runs) > 1:
        with st.expander("📊 Full Parameter Comparison Table", expanded=False):
            display_df = df.drop(columns=["id"])
            # Highlight columns where values differ
            st.dataframe(display_df, width="stretch", hide_index=True)

    return selected_ids


# ── Section 3: Training Curves ────────────────────────────────────────────────

def _render_training_curves(selected_runs: list[dict]) -> None:
    st.markdown("---")
    st.markdown("### :material/show_chart: Training Curves")

    if not any(r.get("log_entries") for r in selected_runs):
        st.warning("No log entries found for the selected runs. Paste training logs when adding a run.")
        return

    # X-axis toggle
    x_axis = st.radio(
        "X-axis",
        ["epoch", "step"],
        horizontal=True,
        key="tc_x_axis",
    )

    # Build dataframes for each metric
    metrics = [
        ("loss", "Loss", "Training Loss vs " + x_axis.title()),
        ("mean_token_accuracy", "Accuracy", "Mean Token Accuracy vs " + x_axis.title()),
        ("entropy", "Entropy", "Entropy vs " + x_axis.title()),
        ("learning_rate", "Learning Rate", "Learning Rate Schedule"),
    ]

    chart_cols = st.columns(2)

    for idx, (metric_key, metric_label, chart_title) in enumerate(metrics):
        # Check if any run has this metric
        has_metric = any(
            any(metric_key in e for e in r.get("log_entries", []))
            for r in selected_runs
        )
        if not has_metric:
            continue

        with chart_cols[idx % 2]:
            st.markdown(f"#### {chart_title}")

            # Build combined dataframe for Altair-style charting
            chart_rows = []
            for run in selected_runs:
                run_name = run.get("name", run["id"])
                for entry in run.get("log_entries", []):
                    if metric_key in entry:
                        x_val = entry.get(x_axis, entry.get("step", 0))
                        chart_rows.append({
                            x_axis: x_val,
                            metric_label: entry[metric_key],
                            "Run": run_name,
                        })

            if chart_rows:
                chart_df = pd.DataFrame(chart_rows)

                # Use st.line_chart with color grouping
                # Pivot for multi-series line chart
                try:
                    pivot_df = chart_df.pivot_table(
                        index=x_axis,
                        columns="Run",
                        values=metric_label,
                        aggfunc="mean",
                    )
                    st.line_chart(pivot_df, width="stretch")
                except Exception:
                    st.line_chart(chart_df, x=x_axis, y=metric_label, color="Run",
                                  width="stretch")


# ── Section 4: Optimal Stopping Analysis ─────────────────────────────────────

def _render_stopping_analysis(selected_runs: list[dict]) -> None:
    st.markdown("---")
    st.markdown("### :material/target: Optimal Stopping Analysis")

    runs_with_logs = [r for r in selected_runs if r.get("log_entries")]
    if not runs_with_logs:
        return

    cols = st.columns(min(len(runs_with_logs), 3))

    for i, run in enumerate(runs_with_logs):
        analysis = analyse_stopping(run.get("log_entries", []))
        if not analysis:
            continue

        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"**{run.get('name', run['id'])}**")

                if "best_loss" in analysis:
                    st.metric(
                        "Best Loss",
                        f"{analysis['best_loss']:.4f}",
                        delta=f"at epoch {analysis['best_loss_epoch']:.2f}"
                              if isinstance(analysis.get('best_loss_epoch'), (int, float))
                              else None,
                        delta_color="off",
                    )

                if "peak_accuracy" in analysis:
                    st.metric(
                        "Peak Accuracy",
                        f"{analysis['peak_accuracy']:.2%}",
                        delta=f"at epoch {analysis['peak_accuracy_epoch']:.2f}"
                              if isinstance(analysis.get('peak_accuracy_epoch'), (int, float))
                              else None,
                        delta_color="off",
                    )

                if "final_loss" in analysis and "best_loss" in analysis:
                    loss_diff = analysis["final_loss"] - analysis["best_loss"]
                    if loss_diff > 0.01:
                        st.metric(
                            "Final vs Best Loss",
                            f"{analysis['final_loss']:.4f}",
                            delta=f"+{loss_diff:.4f}",
                            delta_color="inverse",
                        )

                if analysis.get("overfitting"):
                    st.error(
                        f"⚠️ **Overfitting detected** after epoch "
                        f"{analysis.get('overfit_after_epoch', '?'):.2f} — "
                        f"consider stopping earlier or reducing epochs."
                    )
                elif "best_loss" in analysis:
                    st.success("✅ No overfitting detected — loss trend is healthy.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _short_model_name(full_name: str) -> str:
    """Shorten a HuggingFace model ID for display."""
    if "/" in full_name:
        return full_name.split("/")[-1]
    return full_name
