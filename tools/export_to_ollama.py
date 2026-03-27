#!/usr/bin/env python3
"""
tools/export_to_ollama.py

Loads a trained LoRA adapter (local or from HuggingFace Hub) and exports it
as a GGUF model that can be directly imported into Ollama.

Directory layout:
    models/
    ├── adapters/   # LoRA adapters (local training output + downloaded from HF)
    ├── base/       # Base models (downloaded once, reused across exports)
    └── gguf/       # Merged GGUF output (for Ollama import)

Supports:
  - Local adapter path (default: ./models/adapters/my-lora)
  - HuggingFace model ID (e.g. unsloth/Ministral-3-8B-Instruct-2512-bnb-4bit)
  - Specific training checkpoint (e.g. checkpoint-400)

Usage:
    # From a local adapter
    python tools/export_to_ollama.py --adapter ./models/adapters/my-lora

    # From a HuggingFace model (downloads only the tensors)
    python tools/export_to_ollama.py --base-model unsloth/Ministral-3-8B-Instruct-2512-bnb-4bit

    # From a specific checkpoint
    python tools/export_to_ollama.py --adapter ./models/adapters/my-lora --checkpoint checkpoint-400

    # Full pipeline with Ollama registration
    python tools/export_to_ollama.py \\
        --adapter ./models/adapters/my-lora --checkpoint checkpoint-400 \\
        --ollama-name my-model --quant q4_k_m
"""

import shutil
import sys
import subprocess
import tempfile
from pathlib import Path

# ── Directory constants ───────────────────────────────────────────────────────

ADAPTERS_DIR = Path("models/adapters")
BASE_DIR = Path("models/base")
GGUF_DIR = Path("models/gguf")

# Legacy paths for backward compatibility
_LEGACY_MODELS_DIR = Path("models")
_LEGACY_HF_CACHE = Path("models/_hf_cache")


def _try_download_gguf(model_id: str, quant_method: str, out_path: str) -> str | None:
    """Try to find and download a pre-made GGUF from HuggingFace.

    Many model authors publish GGUF versions (e.g. ``unsloth/Model-GGUF``).
    If found, downloads the specific quantization file directly — much faster
    than converting from scratch.

    Returns the path to the downloaded GGUF file, or None if not found.
    """
    from huggingface_hub import HfApi, hf_hub_download

    # Common GGUF repo naming patterns
    candidates = [
        f"{model_id}-GGUF",                    # unsloth/Model → unsloth/Model-GGUF
        f"{model_id}-gguf",                    # lowercase variant
    ]

    # Also try stripping BNB suffix first
    import re
    clean_id = re.sub(r"-bnb-\d+bit$", "", model_id)
    if clean_id != model_id:
        candidates.insert(0, f"{clean_id}-GGUF")
        candidates.insert(1, f"{clean_id}-gguf")

    api = HfApi()
    for repo_id in candidates:
        try:
            # Check if the repo exists and has GGUF files
            files = api.list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if not gguf_files:
                continue

            # Find the best matching quantization file
            quant_lower = quant_method.lower().replace("_", "-")
            match = None
            for f in gguf_files:
                if quant_lower in f.lower().replace("_", "-"):
                    match = f
                    break
            if not match:
                # Fallback to first GGUF file
                match = gguf_files[0]

            print(f"  Found pre-made GGUF: {repo_id}/{match}")
            output_dir = Path(out_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=match,
                local_dir=str(output_dir),
            )
            print(f"  Downloaded to: {local_path}")

            # Create Modelfile with relative path to the GGUF file
            gguf_filename = Path(local_path).name
            modelfile_path = output_dir / "Modelfile"
            modelfile_path.write_text(f"FROM ./{gguf_filename}\n")
            print(f"  Modelfile created: {modelfile_path}")

            return str(output_dir)
        except Exception:
            continue

    return None


def _strip_bnb_suffix(model_id: str) -> str:
    """Strip BitsAndBytes quantization suffixes from a model ID.

    On MPS/CPU, BNB-quantized models can't be loaded.  This resolves
    e.g. ``unsloth/Ministral-3-8B-Instruct-2512-bnb-4bit``
    → ``unsloth/Ministral-3-8B-Instruct-2512``.
    """
    import re
    cleaned = re.sub(r"-bnb-\d+bit$", "", model_id)
    if cleaned != model_id:
        print(f"  ⚠️  BNB-quantized model detected on MPS/CPU.")
        print(f"       Resolved: {model_id} → {cleaned}")
    return cleaned


def _resolve_base_model_path(base_model_id: str) -> str:
    """Check if a base model exists in models/base/, download if not.

    Returns the local path to the base model directory.
    """
    import re

    # Strip BNB suffix for MPS/CPU compatibility
    import torch
    _has_cuda = torch.cuda.is_available()
    if not _has_cuda:
        resolved_id = _strip_bnb_suffix(base_model_id)
    else:
        resolved_id = base_model_id

    # Create a safe directory name from the model ID
    safe_name = resolved_id.replace("/", "_")
    local_path = BASE_DIR / safe_name

    if local_path.exists() and any(local_path.iterdir()):
        print(f"  Base model found in cache: {local_path}")
        return str(local_path)

    # Download the base model
    print(f"  Downloading base model: {resolved_id} → {local_path}")
    from huggingface_hub import snapshot_download

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=resolved_id,
        local_dir=str(local_path),
    )
    print(f"  Base model cached at: {local_path}")
    return str(local_path)


def _resolve_model_source(
    adapter_path: str,
    base_model: str | None,
    checkpoint: str | None,
) -> str:
    """Determine the model source path or HuggingFace ID."""
    if base_model:
        if checkpoint:
            print(f"Downloading checkpoint '{checkpoint}' from HuggingFace: {base_model}…")
            return _download_hf_checkpoint(base_model, checkpoint)
        else:
            print(f"Model source: HuggingFace {base_model}")
            return base_model
    else:
        model_source = adapter_path
        if checkpoint:
            model_source = str(Path(adapter_path) / checkpoint)
            print(f"Model source: {model_source} (checkpoint)")
        else:
            print(f"Model source: {model_source}")

        if not Path(model_source).exists():
            # Try legacy path (backward compatibility)
            legacy_source = _try_legacy_path(adapter_path)
            if legacy_source:
                model_source = legacy_source
                if checkpoint:
                    model_source = str(Path(legacy_source) / checkpoint)
                print(f"  (resolved from legacy path: {model_source})")
            else:
                print(f"ERROR: Path not found: {model_source}", file=sys.stderr)
                sys.exit(1)
        return model_source


def _try_legacy_path(adapter_path: str) -> str | None:
    """Check legacy model paths for backward compatibility.

    If the adapter_path doesn't exist, try:
    1. models/<name> (old flat layout)
    2. models/_hf_cache/<name> (old HF cache)
    """
    p = Path(adapter_path)
    name = p.name

    # Try old flat layout: models/<name>
    legacy_flat = _LEGACY_MODELS_DIR / name
    if legacy_flat.exists() and legacy_flat.is_dir():
        return str(legacy_flat)

    # Try old HF cache: models/_hf_cache/<name>
    legacy_cache = _LEGACY_HF_CACHE / name
    if legacy_cache.exists() and legacy_cache.is_dir():
        return str(legacy_cache)

    return None


def _download_hf_checkpoint(repo_id: str, checkpoint: str) -> str:
    """Download a specific checkpoint subfolder from a HuggingFace repo.

    Uses ``huggingface_hub.snapshot_download`` to fetch only the files
    inside the checkpoint subfolder (e.g. ``checkpoint-400/``).

    Downloads into models/adapters/ instead of the old _hf_cache location.

    Returns the local path to the downloaded checkpoint directory.
    """
    from huggingface_hub import snapshot_download

    safe_name = repo_id.replace("/", "_")
    local_dir = ADAPTERS_DIR / safe_name
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {repo_id}/{checkpoint} …")
    local_dir_str = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{checkpoint}/*"],
        local_dir=str(local_dir),
    )
    local_checkpoint = str(Path(local_dir_str) / checkpoint)

    if not Path(local_checkpoint).exists():
        # The checkpoint might be at the repo root (not a subfolder)
        # Try downloading the full repo instead
        print(f"  Checkpoint subfolder not found, downloading full repo…")
        local_dir_str = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )
        local_checkpoint = local_dir_str

    print(f"  Downloaded to: {local_checkpoint}")
    return local_checkpoint


def _extract_system_prompt(output_dir: Path, tokenizer) -> str | None:
    """Extract the default system prompt from the model's chat template.

    Checks:
    1. The Jinja2 template for a ``default_system_message`` variable
    2. The tokenizer's chat template string
    3. A ``chat_template.jinja`` file in the output directory

    Returns the system prompt string, or None if not found.
    """
    import re

    # Sources to check for the template
    template_sources = []

    # Check tokenizer's chat_template attribute
    ct = getattr(tokenizer, "chat_template", None)
    if ct and isinstance(ct, str):
        template_sources.append(ct)

    # Check chat_template.jinja file
    jinja_file = output_dir / "chat_template.jinja"
    if jinja_file.exists():
        template_sources.append(jinja_file.read_text(encoding="utf-8"))

    for template in template_sources:
        # Look for: set default_system_message = '...' (handling escaped quotes)
        match = re.search(
            r"set\s+default_system_message\s*=\s*'((?:[^'\\]|\\.)*)'",
            template,
            re.DOTALL,
        )
        if match:
            prompt = match.group(1)
            # Unescape Jinja escapes
            prompt = prompt.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
            return prompt.strip()

    return None


def _detect_quantization_backend() -> str:
    """Detect the best available GGUF quantization backend.

    Checks in order of preference:
    1. ``llama-cpp-python`` — Python bindings with ``convert_hf_to_gguf``
    2. ``llama.cpp`` binary — ``llama-quantize`` or ``quantize`` in PATH
    3. ``mlx-lm`` — Apple Silicon native LoRA fusion + GGUF conversion
    4. ``safetensors`` — fallback (no pre-quantization)

    Returns one of: "llama_cpp_python", "llama_cpp_binary", "mlx_lm",
    or "safetensors".
    """
    # 1. Try llama-cpp-python (has convert_hf_to_gguf bundled)
    try:
        from llama_cpp import llama_cpp  # noqa: F401
        return "llama_cpp_python"
    except ImportError:
        pass

    # 2. Try llama.cpp binary tools in PATH
    if shutil.which("llama-quantize") or shutil.which("quantize"):
        return "llama_cpp_binary"

    # 3. Try mlx-lm (Apple Silicon)
    try:
        import mlx_lm  # noqa: F401
        return "mlx_lm"
    except ImportError:
        pass

    return "safetensors"


def _find_convert_script() -> str | None:
    """Locate the ``convert_hf_to_gguf.py`` script.

    Searches:
    1. Bundled with ``llama-cpp-python`` package
    2. In PATH as ``convert_hf_to_gguf``
    3. Common install locations (homebrew, local builds)

    Returns the path to the script, or None if not found.
    """
    # Check if bundled with llama-cpp-python
    try:
        import llama_cpp
        pkg_dir = Path(llama_cpp.__file__).parent
        script = pkg_dir / "scripts" / "convert_hf_to_gguf.py"
        if script.exists():
            return str(script)
    except ImportError:
        pass

    # Check PATH
    if shutil.which("convert_hf_to_gguf"):
        return "convert_hf_to_gguf"
    if shutil.which("convert-hf-to-gguf"):
        return "convert-hf-to-gguf"

    # Check common locations
    common_paths = [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py"),
        Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for p in common_paths:
        if p.exists():
            return str(p)

    return None


def _find_quantize_binary() -> str | None:
    """Locate the ``llama-quantize`` (or ``quantize``) binary.

    Returns the binary name/path, or None if not found.
    """
    for name in ("llama-quantize", "quantize"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _convert_safetensors_to_gguf(
    safetensors_dir: Path,
    output_dir: Path,
    quant_method: str,
    backend: str,
) -> str | None:
    """Convert a safetensors model directory to a quantized GGUF file.

    Uses either ``llama-cpp-python`` (which bundles ``convert_hf_to_gguf.py``)
    or standalone ``llama.cpp`` binaries.

    Steps:
    1. Convert safetensors → f16 GGUF using ``convert_hf_to_gguf.py``
    2. Quantize f16 GGUF → target quant using ``llama-quantize``
    3. Clean up intermediate f16 GGUF

    Returns the path to the quantized GGUF file, or None on failure.
    """
    convert_script = _find_convert_script()
    quantize_bin = _find_quantize_binary()

    if not convert_script:
        print("  ⚠️  convert_hf_to_gguf.py not found — cannot convert to GGUF")
        return None

    # Step 1: Convert safetensors → f16 GGUF
    f16_gguf = output_dir / "model-f16.gguf"
    print(f"\n  Converting safetensors → f16 GGUF…")
    print(f"    Script: {convert_script}")

    # Build the convert command
    if convert_script.endswith(".py"):
        convert_cmd = [sys.executable, convert_script]
    else:
        convert_cmd = [convert_script]

    convert_cmd.extend([
        str(safetensors_dir),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ])

    try:
        result = subprocess.run(
            convert_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            # Print last few lines of output for progress visibility
            lines = result.stdout.strip().split("\n")
            for line in lines[-3:]:
                print(f"    {line}")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ convert_hf_to_gguf failed: {e}")
        if e.stderr:
            for line in e.stderr.strip().split("\n")[-5:]:
                print(f"    {line}")
        return None

    if not f16_gguf.exists():
        print(f"  ❌ Expected f16 GGUF not found: {f16_gguf}")
        return None

    f16_size_gb = f16_gguf.stat().st_size / (1024 ** 3)
    print(f"    f16 GGUF created: {f16_size_gb:.1f} GB")

    # Step 2: Quantize if not f16
    if quant_method.lower() == "f16":
        final_gguf = output_dir / "model-f16.gguf"
        print(f"  ✅ f16 requested — no quantization needed.")
        return str(final_gguf)

    if not quantize_bin:
        print(f"  ⚠️  llama-quantize not found — keeping f16 GGUF")
        print(f"       Install llama.cpp to enable quantization:")
        print(f"       brew install llama.cpp  (macOS)")
        # Rename to indicate it's f16
        return str(f16_gguf)

    quant_upper = quant_method.upper()
    final_gguf = output_dir / f"model-{quant_upper}.gguf"
    print(f"\n  Quantizing f16 → {quant_upper}…")
    print(f"    Binary: {quantize_bin}")

    try:
        result = subprocess.run(
            [quantize_bin, str(f16_gguf), str(final_gguf), quant_upper],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines[-3:]:
                print(f"    {line}")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ llama-quantize failed: {e}")
        if e.stderr:
            for line in e.stderr.strip().split("\n")[-5:]:
                print(f"    {line}")
        print(f"  Falling back to f16 GGUF.")
        return str(f16_gguf)

    if not final_gguf.exists():
        print(f"  ❌ Expected quantized GGUF not found: {final_gguf}")
        return str(f16_gguf)

    final_size_gb = final_gguf.stat().st_size / (1024 ** 3)
    print(f"    Quantized GGUF created: {final_size_gb:.1f} GB")

    # Clean up intermediate f16 GGUF (it's large)
    try:
        f16_gguf.unlink()
        print(f"    Cleaned up intermediate f16 GGUF")
    except OSError:
        pass

    return str(final_gguf)


def _convert_with_mlx(
    model_source: str,
    output_dir: Path,
    quant_method: str,
    is_lora: bool,
    adapter_path: str | None = None,
) -> str | None:
    """Convert a model to GGUF using mlx-lm (Apple Silicon native).

    For LoRA adapters, uses ``mlx_lm.fuse`` to merge adapter + base,
    then converts to GGUF.  For full models, converts directly.

    Returns the path to the GGUF file, or None on failure.
    """
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return None

    print(f"\n  Using mlx-lm for Apple Silicon native conversion…")

    # mlx-lm works best with its own conversion pipeline
    # Step 1: If LoRA, fuse the adapter first
    fused_dir = output_dir / "_mlx_fused"

    if is_lora and adapter_path:
        print(f"  Fusing LoRA adapter with base model…")
        fuse_cmd = [
            sys.executable, "-m", "mlx_lm.fuse",
            "--model", model_source,
            "--adapter-path", adapter_path,
            "--save-path", str(fused_dir),
        ]
        try:
            subprocess.run(fuse_cmd, check=True, capture_output=True, text=True)
            print(f"    Fused model saved to: {fused_dir}")
            convert_source = str(fused_dir)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ mlx_lm.fuse failed: {e}")
            if e.stderr:
                for line in e.stderr.strip().split("\n")[-5:]:
                    print(f"    {line}")
            return None
    else:
        convert_source = model_source

    # Step 2: Convert to GGUF using mlx_lm.convert
    # mlx-lm can convert to GGUF format with quantization
    quant_upper = quant_method.upper()
    gguf_path = output_dir / f"model-{quant_upper}.gguf"

    # Map common quant names to mlx-lm compatible formats
    mlx_quant_map = {
        "Q4_K_M": "4",
        "Q4_0": "4",
        "Q8_0": "8",
        "F16": "16",
        "Q4_K_S": "4",
        "Q5_K_M": "4",  # mlx-lm uses simpler bit widths
        "Q5_K_S": "4",
        "Q6_K": "8",
    }
    mlx_bits = mlx_quant_map.get(quant_upper, "4")

    convert_cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--model", convert_source,
        "--quantize",
        "--q-bits", mlx_bits,
        "--upload-repo", "",  # empty = don't upload
    ]

    # mlx_lm.convert outputs to mlx format, not GGUF directly.
    # We need to use the safetensors → GGUF path instead.
    # Fall back to saving safetensors from mlx and converting via llama.cpp.
    print(f"  ⚠️  mlx-lm does not produce GGUF directly.")
    print(f"       Saving merged model as safetensors, then converting…")

    # Clean up fused dir if we created it
    if fused_dir.exists():
        import shutil as _shutil
        _shutil.rmtree(str(fused_dir), ignore_errors=True)

    return None


def _export_hf_fallback(model_source: str, out_path: str, quant_method: str) -> tuple[str, bool]:
    """Export using HuggingFace transformers (MPS/CPU fallback).

    Loads the model with transformers, merges LoRA if present, and attempts
    to produce a pre-quantized GGUF file using the best available backend.

    Backend priority:
    1. ``llama-cpp-python`` / ``llama.cpp`` binary → GGUF conversion + quantization
    2. ``mlx-lm`` → Apple Silicon native (LoRA fusion + conversion)
    3. Safetensors fallback → Ollama quantizes during import

    For LoRA adapters, reads adapter_config.json to find the base model,
    checks models/base/ cache, and downloads if needed.

    Returns a tuple of (output_directory_path, is_pre_quantized_gguf).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Mistral3 only ships ForConditionalGeneration; register it so
    # AutoModelForCausalLM.from_pretrained works.
    try:
        from transformers.models.mistral3 import (
            Mistral3Config,
            Mistral3ForConditionalGeneration,
        )
        AutoModelForCausalLM.register(Mistral3Config, Mistral3ForConditionalGeneration)
    except (ImportError, AttributeError):
        pass

    output_dir = Path(out_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect available quantization backend
    backend = _detect_quantization_backend()
    print(f"  Quantization backend: {backend}")

    # Detect dtype
    _has_mps = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    dtype = torch.bfloat16 if _has_mps else torch.float32

    # Check if this is a LoRA adapter (has adapter_config.json)
    source_path = Path(model_source)
    is_lora = (source_path.exists()
               and (source_path / "adapter_config.json").exists())

    if is_lora:
        print("  Detected LoRA adapter — merging with base model…")
        from peft import PeftModel, PeftConfig

        # Load the adapter config to find the base model
        peft_config = PeftConfig.from_pretrained(model_source)
        base_model_id = peft_config.base_model_name_or_path

        # Resolve base model: check models/base/ cache, download if needed
        base_model_path = _resolve_base_model_path(base_model_id)
        print(f"  Base model: {base_model_path}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_source)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True,
        )
    else:
        # Strip BNB quantization suffix for full model loading on MPS/CPU
        resolved_source = _strip_bnb_suffix(model_source)
        print(f"  Loading full model: {resolved_source}…")
        model = AutoModelForCausalLM.from_pretrained(
            resolved_source,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_source, trust_remote_code=True,
        )

    print(f"  Model loaded: {model.config.model_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Save merged model as safetensors (needed for all paths)
    print(f"\nSaving merged model to {output_dir}…")
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    # Free model memory before quantization
    del model
    try:
        import gc
        gc.collect()
        if _has_mps:
            torch.mps.empty_cache()
    except Exception:
        pass

    # ── Attempt GGUF conversion ───────────────────────────────────────────
    gguf_path = None
    is_pre_quantized = False

    if backend in ("llama_cpp_python", "llama_cpp_binary"):
        print(f"\n  🔧 Using llama.cpp for GGUF conversion + quantization…")
        gguf_path = _convert_safetensors_to_gguf(
            safetensors_dir=output_dir,
            output_dir=output_dir,
            quant_method=quant_method,
            backend=backend,
        )
        if gguf_path:
            is_pre_quantized = True

    elif backend == "mlx_lm":
        # Try mlx-lm first; it may fall back to None
        gguf_path = _convert_with_mlx(
            model_source=model_source,
            output_dir=output_dir,
            quant_method=quant_method,
            is_lora=is_lora,
            adapter_path=model_source if is_lora else None,
        )
        if gguf_path:
            is_pre_quantized = True
        else:
            # mlx-lm couldn't produce GGUF; try llama.cpp as secondary
            convert_script = _find_convert_script()
            if convert_script:
                print(f"\n  🔧 Falling back to llama.cpp for GGUF conversion…")
                gguf_path = _convert_safetensors_to_gguf(
                    safetensors_dir=output_dir,
                    output_dir=output_dir,
                    quant_method=quant_method,
                    backend="llama_cpp_binary",
                )
                if gguf_path:
                    is_pre_quantized = True

    # ── Build Modelfile ───────────────────────────────────────────────────
    if is_pre_quantized and gguf_path:
        gguf_filename = Path(gguf_path).name
        modelfile_content = f"FROM ./{gguf_filename}\n"
        print(f"\n  ✅ Pre-quantized GGUF: {gguf_filename}")

        # Clean up safetensors files (they're no longer needed)
        _cleanup_safetensors(output_dir, keep_gguf=gguf_filename)
    else:
        # Fallback: Ollama will import safetensors and quantize during create
        modelfile_content = "FROM .\n"
        if quant_method != "f16":
            print(f"\n  ⚠️  No GGUF backend available.")
            print(f"       Quantization ({quant_method}) will be applied by Ollama during import.")
            print(f"       Install llama.cpp for faster pre-quantized exports:")
            print(f"         brew install llama.cpp")
        else:
            print(f"\n  Full precision (f16) — no quantization.")

    # Extract default system prompt from the chat template (Jinja2)
    system_prompt = _extract_system_prompt(output_dir, tokenizer)
    if system_prompt:
        # Escape triple quotes for Modelfile
        escaped = system_prompt.replace('"""', '\\"\\"\\"')
        modelfile_content += f'\nSYSTEM """{escaped}"""\n'
        print(f"  System prompt: {system_prompt[:80]}…")

    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(modelfile_content)
    print(f"  Modelfile created: {modelfile_path}")

    return str(output_dir), is_pre_quantized


def _cleanup_safetensors(output_dir: Path, keep_gguf: str) -> None:
    """Remove safetensors and related files after successful GGUF conversion.

    Keeps the GGUF file, Modelfile, chat_template.jinja, and tokenizer config.
    Removes large safetensors files to save disk space.
    """
    # Files to always keep
    keep_patterns = {
        keep_gguf,
        "Modelfile",
        "chat_template.jinja",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "config.json",
    }

    removed_size = 0
    for f in output_dir.iterdir():
        if f.name in keep_patterns:
            continue
        if f.suffix in (".safetensors", ".bin") or f.name.startswith("model-00"):
            size = f.stat().st_size
            f.unlink()
            removed_size += size

    if removed_size > 0:
        print(f"  Cleaned up {removed_size / (1024**3):.1f} GB of safetensors files")


def export_model(
    adapter_path: str = "./models/adapters/my-lora",
    out_path: str = "./models/gguf/my-lora",
    quant_method: str = "q4_k_m",
    ollama_name: str | None = None,
    base_model: str | None = None,
    checkpoint: str | None = None,
):
    """Export a LoRA adapter or HuggingFace model to GGUF for Ollama.

    Pipeline:
    1. Accept adapter path (from models/adapters/)
    2. Read adapter_config.json to find base model ID
    3. Check if base model exists in models/base/ — if not, download it
    4. Merge adapter + base → save GGUF to models/gguf/name/
    5. Create Modelfile with FROM ./model.gguf
    6. Run ollama create

    Parameters
    ----------
    adapter_path : str
        Path to the local LoRA adapter directory.
    out_path : str
        Output directory for the GGUF file (default: models/gguf/).
    quant_method : str
        Quantization method (e.g. q4_k_m, q8_0, f16).
    ollama_name : str or None
        If provided, register the model in Ollama under this name.
    base_model : str or None
        HuggingFace model ID to download and export directly.
        When provided, ``adapter_path`` is ignored.
    checkpoint : str or None
        Checkpoint subdirectory name (e.g. "checkpoint-400").
        Appended to ``adapter_path`` to load a specific checkpoint.
    """
    # Ensure output directories exist
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    # Determine the model source
    model_source = _resolve_model_source(adapter_path, base_model, checkpoint)

    # Try Unsloth first (NVIDIA/AMD/Intel GPUs), fall back to HuggingFace
    # for Apple Silicon (MPS) and CPU-only systems.
    _use_unsloth = False
    _is_pre_quantized = False
    try:
        from unsloth import FastLanguageModel
        _use_unsloth = True
    except Exception:
        pass

    if _use_unsloth:
        print(f"Loading model with Unsloth: {model_source}…")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_source,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )

        print(f"  Model loaded: {model.config.model_type}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        print(f"\nExporting to GGUF format ({quant_method})…")
        print("This will take a few minutes as it merges the LoRA and quantizes.")

        model.save_pretrained_gguf(out_path, tokenizer, quantization_method=quant_method)
        gguf_dir = f"{out_path}_gguf"
    else:
        # MPS / CPU fallback.
        # Step 1: Try to find a pre-made GGUF on HuggingFace (fastest path).
        gguf_dir = None
        _is_pre_quantized = False
        if base_model and not checkpoint:
            print(f"Searching for pre-made GGUF on HuggingFace…")
            gguf_dir = _try_download_gguf(model_source, quant_method, out_path)
            if gguf_dir:
                _is_pre_quantized = True  # Downloaded GGUF is already quantized

        # Step 2: If no GGUF found, load with transformers + merge + convert.
        if not gguf_dir:
            print(f"Loading model with HuggingFace (MPS/CPU fallback): {model_source}…")
            gguf_dir, _is_pre_quantized = _export_hf_fallback(
                model_source, out_path, quant_method,
            )

    print(f"\n✅ Export complete! Output files are in: {gguf_dir}")

    # Import into Ollama if requested
    if ollama_name:
        print(f"\nRegistering model '{ollama_name}' in Ollama…")
        # Build the ollama create command
        cmd = ["ollama", "create", ollama_name, "-f", "Modelfile"]

        # Add quantization flag ONLY for safetensors imports (no pre-quantized GGUF).
        # When we already have a quantized GGUF (from Unsloth, llama.cpp, or
        # a downloaded GGUF), Ollama imports it directly — no --quantize needed.
        _needs_ollama_quant = (
            not _use_unsloth
            and not _is_pre_quantized
            and quant_method != "f16"
        )
        if _needs_ollama_quant:
            # Ollama expects uppercase format: q4_K_M, q8_0, etc.
            ollama_quant = quant_method.upper().replace("_K_", "_K_")
            cmd.extend(["--quantize", ollama_quant])
            print(f"  Quantization: {ollama_quant} (Ollama will quantize during import)")
        elif not _use_unsloth and _is_pre_quantized:
            print(f"  Using pre-quantized GGUF — no Ollama quantization needed")

        try:
            subprocess.run(cmd, cwd=gguf_dir, check=True)
            print(f"✅ Successfully compiled '{ollama_name}' into Ollama!")
            print(f"To run it: ollama run {ollama_name}")
        except Exception as e:
            print(f"❌ Failed to run 'ollama create': {e}")
            print(f"You can manually run: cd {gguf_dir} && {' '.join(cmd)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export LoRA adapter or HuggingFace model to Ollama GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From a local adapter
  python tools/export_to_ollama.py --adapter ./models/adapters/my-lora

  # From a specific checkpoint
  python tools/export_to_ollama.py --adapter ./models/adapters/my-lora --checkpoint checkpoint-400

  # From a HuggingFace model ID
  python tools/export_to_ollama.py --base-model unsloth/Ministral-3-8B-Instruct-2512-bnb-4bit

  # Full pipeline
  python tools/export_to_ollama.py \\
      --adapter ./models/adapters/my-lora --checkpoint checkpoint-400 \\
      --ollama-name my-model --quant q4_k_m
        """,
    )
    parser.add_argument(
        "--adapter", default="./models/adapters/my-lora",
        help="Path to your trained LoRA adapter (default: ./models/adapters/my-lora)",
    )
    parser.add_argument(
        "--base-model", default=None,
        help="HuggingFace model ID to download and export directly. "
             "When provided, --adapter is ignored.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint subdirectory (e.g. 'checkpoint-400'). "
             "Appended to --adapter path.",
    )
    parser.add_argument(
        "--output", default="./models/gguf/my-lora",
        help="Output directory for GGUF file (default: ./models/gguf/my-lora)",
    )
    parser.add_argument(
        "--quant", default="q4_k_m",
        help="Quantization method (e.g. q4_k_m, q8_0, f16). Default: q4_k_m",
    )
    parser.add_argument(
        "--ollama-name", default=None,
        help="If provided, automatically run 'ollama create' with this name.",
    )
    args = parser.parse_args()

    export_model(
        adapter_path=args.adapter,
        out_path=args.output,
        quant_method=args.quant,
        ollama_name=args.ollama_name,
        base_model=args.base_model,
        checkpoint=args.checkpoint,
    )
