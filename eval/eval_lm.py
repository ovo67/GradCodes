"""
Custom lm-eval-harness model wrapper for QLoRA models with multi-GPU support

This wrapper loads the base model with optional quantization (4-bit NF4 or 16-bit BF16)
and applies LoRA adapters with optional quantization (4-bit or 16-bit) for evaluation.
It can also apply a saved Gradcodes artifact, Gradcodes/PV-Tuning search-state
checkpoint, QZO/QuZO/QES checkpoint, LoQT checkpoint, or QA-LoRA checkpoint
directly at evaluation time.

Usage (single GPU - base model only with 4-bit):
    CUDA_VISIBLE_DEVICES=0 python eval_lm.py \\
        --base_model meta-llama/Llama-3.2-1B-Instruct \\
        --base_bits 4 \\
        --tasks mmlu

Usage (single GPU - with LoRA, base 4-bit, adapter 16-bit):
    CUDA_VISIBLE_DEVICES=0 python eval_lm.py \\
        --base_model meta-llama/Llama-3.2-1B-Instruct \\
        --lora_model ./llama-3.2-1b-qlora-alpaca \\
        --base_bits 4 \\
        --adapter_bits 16 \\
        --tasks mmlu

Usage (multi-GPU with accelerate):
    CUDA_VISIBLE_DEVICES=1,2 accelerate launch \\
        --main_process_port 11011 \\
        --num_processes 2 \\
        eval_lm.py \\
        --base_model meta-llama/Llama-3.2-1B-Instruct \\
        --lora_model ./llama-3.2-1b-qlora-alpaca \\
        --base_bits 4 \\
        --adapter_bits 16 \\
        --tasks mmlu \\
        --batch_size 4

    CUDA_VISIBLE_DEVICES=1,2 accelerate launch \\
        --main_process_port 11011 \\
        --num_processes 2 \\
        eval_lm.py \\
        --base_model meta-llama/Llama-3.2-1B-Instruct \\
        --gradcodes_artifact ./outputs/llama-3.2-1b-gradcodes-gsm8k \\
        --base_bits 16 \\
        --tasks gsm8k_cot_zeroshot \\
        --batch_size 32

    CUDA_VISIBLE_DEVICES=1,2 accelerate launch \\
        --main_process_port 11011 \\
        --num_processes 2 \\
        eval_lm.py \\
        --base_model meta-llama/Llama-3.2-1B-Instruct \\
        --pvtuning_artifact ./outputs/llama-3.2-1b-pvtuning-ds-gsm8k/epoch_0001 \\
        --base_bits 4 \\
        --tasks gsm8k_cot_zeroshot \\
        --batch_size 32

    CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \
        --main_process_port 11011 \
        --num_processes 4 \
        eval_lm.py \
        --base_model meta-llama/Llama-3.2-1B-Instruct \
        --qzo_artifact ./outputs/llama-3.2-1b-qzo-gsm8k/epoch_0002 \
        --base_bits 4 \
        --tasks gsm8k_cot_zeroshot \
        --batch_size 32

    CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \
        --main_process_port 11011 \
        --num_processes 4 \
        eval_lm.py \
        --base_model meta-llama/Llama-3.2-1B-Instruct \
        --quzo_artifact ./outputs/llama-3.2-1b-quzo-gsm8k/epoch_0003 \
        --base_bits 4 \
        --tasks gsm8k_cot_zeroshot \
        --batch_size 32

    CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \\
        --main_process_port 11011 \\
        --num_processes 4 \\
        eval_lm.py \\
        --base_model meta-llama/Llama-3.2-1B-Instruct \\
        --qes_artifact ./outputs/llama-3.2-1b-qes-gsm8k/epoch_0001 \\
        --base_bits 4 \\
        --tasks gsm8k_cot_zeroshot \\
        --batch_size 32
"""
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Set Hugging Face mirror for faster downloads
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def configure_default_hf_cache_dirs() -> None:
    """Use the standard Hugging Face cache location or user-supplied cache variables."""
    return None


configure_default_hf_cache_dirs()

import torch
import torch.nn as nn
from typing import Optional, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

apply_gradcodes_artifact_to_model = None
load_gradcodes_artifact = None

try:
    from src.gradcodes import apply_search_state_to_model
except ImportError:
    apply_search_state_to_model = None

try:
    from loqt import apply_loqt_state_to_model, iter_loqt_modules
except ImportError:
    apply_loqt_state_to_model = None
    iter_loqt_modules = None

try:
    from qalora import apply_qalora_state_to_model
except ImportError:
    apply_qalora_state_to_model = None

try:
    from qzo import apply_qzo_state_to_model
except ImportError:
    apply_qzo_state_to_model = None

try:
    from quzo import apply_quzo_state_to_model
except ImportError:
    apply_quzo_state_to_model = None

try:
    from qes import apply_qes_state_to_model
except ImportError:
    apply_qes_state_to_model = None


def setup_distributed():
    """Setup distributed training if using accelerate"""
    try:
        from accelerate import PartialState
        distributed_state = PartialState()
        return distributed_state
    except ImportError:
        return None


def get_compute_dtype(bits: int) -> torch.dtype:
    """Map bit-width choice to a torch compute dtype."""
    if bits == 32:
        return torch.float32
    return torch.bfloat16


def torch_dtype_from_name(name: Optional[str]) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype value in checkpoint metadata: {name}")
    return mapping[name]


def is_gradcodes_artifact_dir(path: Optional[str]) -> bool:
    if not path:
        return False
    artifact_dir = Path(path)
    return artifact_dir.is_dir() and (artifact_dir / "artifact_config.json").exists() and (artifact_dir / "gradcodes_state.pt").exists()


def resolve_gradcodes_search_state_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_file() and candidate.name == "gradcodes_state.pt":
        return candidate
    if candidate.is_dir():
        state_file = candidate / "gradcodes_state.pt"
        if state_file.exists():
            return state_file
    return None


def resolve_loqt_state_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_file() and candidate.name == "loqt_state.pt":
        return candidate
    if candidate.is_dir():
        state_file = candidate / "loqt_state.pt"
        if state_file.exists():
            return state_file
    return None


def resolve_qalora_state_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_file() and candidate.name == "qalora_state.pt":
        return candidate
    if candidate.is_dir():
        state_file = candidate / "qalora_state.pt"
        if state_file.exists():
            return state_file
    return None


def resolve_quzo_state_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_file() and candidate.name == "quzo_state.pt":
        return candidate
    if candidate.is_dir():
        state_file = candidate / "quzo_state.pt"
        if state_file.exists():
            return state_file
    return None


def resolve_qzo_state_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_file() and candidate.name == "qzo_state.pt":
        return candidate
    if candidate.is_dir():
        state_file = candidate / "qzo_state.pt"
        if state_file.exists():
            return state_file
    return None


def resolve_qes_state_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_file() and candidate.name == "qes_state.pt":
        return candidate
    if candidate.is_dir():
        state_file = candidate / "qes_state.pt"
        if state_file.exists():
            return state_file
    return None


def load_gradcodes_search_checkpoint(path: str) -> dict:
    state_file = resolve_gradcodes_search_state_path(path)
    if state_file is None:
        raise ValueError(f"{path} does not point to a gradcodes_state.pt checkpoint.")

    payload = torch.load(state_file, map_location="cpu")
    if not isinstance(payload, dict) or "search_state" not in payload:
        raise ValueError(f"{state_file} is not a valid Gradcodes search-state checkpoint.")
    return payload


def summarize_gradcodes_search_quantization(payload: dict) -> dict:
    search_state = payload.get("search_state", {})
    quant_types = sorted({state.get("quant_type") for state in search_state.values()})
    quant_bits = sorted({int(state.get("bits")) for state in search_state.values()})
    group_sizes = sorted({int(state.get("group_size")) for state in search_state.values()})
    return {
        "quant_types": quant_types,
        "quant_bits": quant_bits,
        "group_sizes": group_sizes,
    }


def is_pvtuning_search_checkpoint_payload(payload: dict) -> bool:
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    output_dir = str(args.get("output_dir", "")).lower()
    return (
        "pv_update_fraction" in args
        or "pv_min_updates" in args
        or "pvtuning" in output_dir
        or "pv-tuning" in output_dir
    )


def load_loqt_checkpoint(path: str) -> dict:
    state_file = resolve_loqt_state_path(path)
    if state_file is None:
        raise ValueError(f"{path} does not point to a loqt_state.pt checkpoint.")

    payload = torch.load(state_file, map_location="cpu")
    if not isinstance(payload, dict) or "loqt_state" not in payload:
        raise ValueError(f"{state_file} is not a valid LoQT checkpoint.")
    return payload


def summarize_loqt_quantization(payload: dict) -> dict:
    loqt_state = payload.get("loqt_state", {})
    quant_types = sorted({state.get("quant_type") for state in loqt_state.values()})
    quant_bits = sorted({int(state.get("bits")) for state in loqt_state.values()})
    group_sizes = sorted({int(state.get("group_size")) for state in loqt_state.values()})
    ranks = sorted({int(state.get("rank")) for state in loqt_state.values()})
    return {
        "quant_types": quant_types,
        "quant_bits": quant_bits,
        "group_sizes": group_sizes,
        "ranks": ranks,
    }


def load_qalora_checkpoint(path: str) -> dict:
    state_file = resolve_qalora_state_path(path)
    if state_file is None:
        raise ValueError(f"{path} does not point to a qalora_state.pt checkpoint.")

    payload = torch.load(state_file, map_location="cpu")
    if not isinstance(payload, dict) or "qalora_state" not in payload:
        raise ValueError(f"{state_file} is not a valid QA-LoRA checkpoint.")
    return payload


def summarize_qalora_quantization(payload: dict) -> dict:
    qalora_state = payload.get("qalora_state", {})
    quant_types = sorted({state.get("quant_type") for state in qalora_state.values()})
    quant_bits = sorted({int(state.get("bits")) for state in qalora_state.values()})
    group_sizes = sorted({int(state.get("group_size")) for state in qalora_state.values()})
    ranks = sorted({int(state.get("rank")) for state in qalora_state.values()})
    adapter_group_sizes = sorted({int(state.get("adapter_group_size")) for state in qalora_state.values()})
    return {
        "quant_types": quant_types,
        "quant_bits": quant_bits,
        "group_sizes": group_sizes,
        "ranks": ranks,
        "adapter_group_sizes": adapter_group_sizes,
    }


def load_quzo_checkpoint(path: str) -> dict:
    state_file = resolve_quzo_state_path(path)
    if state_file is None:
        raise ValueError(f"{path} does not point to a quzo_state.pt checkpoint.")

    payload = torch.load(state_file, map_location="cpu")
    if not isinstance(payload, dict) or "search_state" not in payload:
        raise ValueError(f"{state_file} is not a valid QuZO checkpoint.")
    return payload


def summarize_quzo_quantization(payload: dict) -> dict:
    search_state = payload.get("search_state", {})
    quant_types = sorted({state.get("quant_type") for state in search_state.values()})
    quant_bits = sorted({int(state.get("bits")) for state in search_state.values()})
    group_sizes = sorted({int(state.get("group_size")) for state in search_state.values()})
    return {
        "quant_types": quant_types,
        "quant_bits": quant_bits,
        "group_sizes": group_sizes,
    }


def load_qzo_checkpoint(path: str) -> dict:
    state_file = resolve_qzo_state_path(path)
    if state_file is None:
        raise ValueError(f"{path} does not point to a qzo_state.pt checkpoint.")

    payload = torch.load(state_file, map_location="cpu")
    if not isinstance(payload, dict) or "search_state" not in payload:
        raise ValueError(f"{state_file} is not a valid QZO checkpoint.")
    return payload


def summarize_qzo_quantization(payload: dict) -> dict:
    search_state = payload.get("search_state", {})
    quant_types = sorted({state.get("quant_type") for state in search_state.values()})
    quant_bits = sorted({int(state.get("bits")) for state in search_state.values()})
    group_sizes = sorted({int(state.get("group_size")) for state in search_state.values()})
    return {
        "quant_types": quant_types,
        "quant_bits": quant_bits,
        "group_sizes": group_sizes,
    }


def load_qes_checkpoint(path: str) -> dict:
    state_file = resolve_qes_state_path(path)
    if state_file is None:
        raise ValueError(f"{path} does not point to a qes_state.pt checkpoint.")

    payload = torch.load(state_file, map_location="cpu")
    if not isinstance(payload, dict) or "search_state" not in payload:
        raise ValueError(f"{state_file} is not a valid QES checkpoint.")
    return payload


def summarize_qes_quantization(payload: dict) -> dict:
    search_state = payload.get("search_state", {})
    quant_types = sorted({state.get("quant_type") for state in search_state.values()})
    quant_bits = sorted({int(state.get("bits")) for state in search_state.values()})
    group_sizes = sorted({int(state.get("group_size")) for state in search_state.values()})
    return {
        "quant_types": quant_types,
        "quant_bits": quant_bits,
        "group_sizes": group_sizes,
    }


def resolve_gradcodes_artifact(
    base_model_path: str,
    gradcodes_artifact_path: Optional[str],
    *,
    search_state_format: str = "gradcodes_search_state",
    search_state_label: str = "Gradcodes",
) -> tuple[str, Optional[str], Optional[dict], Optional[dict], Optional[str]]:
    """
    Resolve the effective base model path and optional Gradcodes input path.

    Returns:
        effective_base_model_path, gradcodes_path, metadata, search_checkpoint_payload, tokenizer_source
    """
    artifact_path = gradcodes_artifact_path
    base_model_is_old_artifact = is_gradcodes_artifact_dir(base_model_path)
    base_model_search_state = resolve_gradcodes_search_state_path(base_model_path)
    if artifact_path is None and (base_model_is_old_artifact or base_model_search_state is not None):
        artifact_path = base_model_path

    if artifact_path is None:
        return base_model_path, None, None, None, base_model_path

    search_state_path = resolve_gradcodes_search_state_path(artifact_path)
    if search_state_path is not None:
        payload = load_gradcodes_search_checkpoint(str(search_state_path))
        if base_model_is_old_artifact or base_model_search_state is not None:
            effective_base_model = payload.get("base_model_name_or_path")
        else:
            effective_base_model = base_model_path

        if not effective_base_model:
            raise ValueError(
                "Could not resolve the base model path for the Gradcodes search-state checkpoint. "
                "Pass --base_model explicitly or ensure the checkpoint metadata contains it."
            )

        tokenizer_root = search_state_path.parent
        tokenizer_source = str(tokenizer_root)
        if not ((tokenizer_root / "tokenizer.json").exists() or (tokenizer_root / "tokenizer_config.json").exists()):
            tokenizer_source = effective_base_model

        quant_summary = summarize_gradcodes_search_quantization(payload)
        metadata = {
            "format": search_state_format,
            "state_label": search_state_label,
            "num_modules": len(payload.get("search_state", {})),
            "checkpoint_label": payload.get("checkpoint_label"),
            "quant_types": quant_summary["quant_types"],
            "quant_bits": quant_summary["quant_bits"],
            "group_sizes": quant_summary["group_sizes"],
            "train_torch_dtype": payload.get("args", {}).get("torch_dtype"),
        }
        return effective_base_model, str(search_state_path), metadata, payload, tokenizer_source

    if load_gradcodes_artifact is None:
        raise ImportError(
            "This artifact format is not bundled; use a Gradcodes search-state checkpoint instead."
        )

    metadata, _ = load_gradcodes_artifact(artifact_path)

    if base_model_is_old_artifact:
        effective_base_model = metadata.get("resolved_model_path") or metadata.get("base_model_name_or_path")
    else:
        effective_base_model = base_model_path

    if not effective_base_model:
        raise ValueError(
            "Could not resolve the base model path for the Gradcodes artifact. "
            "Pass --base_model explicitly or ensure the artifact metadata contains it."
        )

    tokenizer_source = artifact_path
    if not ((Path(artifact_path) / "tokenizer.json").exists() or (Path(artifact_path) / "tokenizer_config.json").exists()):
        tokenizer_source = effective_base_model

    return effective_base_model, artifact_path, metadata, None, tokenizer_source


def resolve_pvtuning_artifact(
    base_model_path: str,
    pvtuning_artifact_path: Optional[str],
) -> tuple[str, Optional[str], Optional[dict], Optional[dict], Optional[str]]:
    return resolve_gradcodes_artifact(
        base_model_path=base_model_path,
        gradcodes_artifact_path=pvtuning_artifact_path,
        search_state_format="pvtuning_ds_search_state",
        search_state_label="PV-Tuning",
    )


def resolve_loqt_artifact(
    base_model_path: str,
    loqt_artifact_path: Optional[str],
) -> tuple[str, Optional[str], Optional[dict], Optional[dict], Optional[str]]:
    """
    Resolve the effective base model path and optional LoQT checkpoint path.

    Returns:
        effective_base_model_path, loqt_path, metadata, loqt_payload, tokenizer_source
    """
    artifact_path = loqt_artifact_path
    base_model_loqt_state = resolve_loqt_state_path(base_model_path)
    if artifact_path is None and base_model_loqt_state is not None:
        artifact_path = base_model_path

    if artifact_path is None:
        return base_model_path, None, None, None, base_model_path

    loqt_state_path = resolve_loqt_state_path(artifact_path)
    if loqt_state_path is None:
        raise ValueError(f"{artifact_path} does not point to a LoQT checkpoint directory or loqt_state.pt file.")

    payload = load_loqt_checkpoint(str(loqt_state_path))
    if base_model_loqt_state is not None:
        effective_base_model = payload.get("base_model_name_or_path")
    else:
        effective_base_model = base_model_path

    if not effective_base_model:
        raise ValueError(
            "Could not resolve the base model path for the LoQT checkpoint. "
            "Pass --base_model explicitly or ensure the checkpoint metadata contains it."
        )

    tokenizer_root = loqt_state_path.parent
    tokenizer_source = str(tokenizer_root)
    if not ((tokenizer_root / "tokenizer.json").exists() or (tokenizer_root / "tokenizer_config.json").exists()):
        tokenizer_source = effective_base_model

    quant_summary = summarize_loqt_quantization(payload)
    metadata = {
        "format": "loqt_state",
        "num_modules": len(payload.get("loqt_state", {})),
        "checkpoint_label": payload.get("checkpoint_label"),
        "quant_types": quant_summary["quant_types"],
        "quant_bits": quant_summary["quant_bits"],
        "group_sizes": quant_summary["group_sizes"],
        "ranks": quant_summary["ranks"],
        "train_torch_dtype": payload.get("args", {}).get("torch_dtype"),
    }
    return effective_base_model, str(loqt_state_path), metadata, payload, tokenizer_source


def resolve_qalora_artifact(
    base_model_path: str,
    qalora_artifact_path: Optional[str],
) -> tuple[str, Optional[str], Optional[dict], Optional[dict], Optional[str]]:
    """
    Resolve the effective base model path and optional QA-LoRA checkpoint path.

    Returns:
        effective_base_model_path, qalora_path, metadata, qalora_payload, tokenizer_source
    """
    artifact_path = qalora_artifact_path
    base_model_qalora_state = resolve_qalora_state_path(base_model_path)
    if artifact_path is None and base_model_qalora_state is not None:
        artifact_path = base_model_path

    if artifact_path is None:
        return base_model_path, None, None, None, base_model_path

    qalora_state_path = resolve_qalora_state_path(artifact_path)
    if qalora_state_path is None:
        raise ValueError(f"{artifact_path} does not point to a QA-LoRA checkpoint directory or qalora_state.pt file.")

    payload = load_qalora_checkpoint(str(qalora_state_path))
    if base_model_qalora_state is not None:
        effective_base_model = payload.get("base_model_name_or_path")
    else:
        effective_base_model = base_model_path

    if not effective_base_model:
        raise ValueError(
            "Could not resolve the base model path for the QA-LoRA checkpoint. "
            "Pass --base_model explicitly or ensure the checkpoint metadata contains it."
        )

    tokenizer_root = qalora_state_path.parent
    tokenizer_source = str(tokenizer_root)
    if not ((tokenizer_root / "tokenizer.json").exists() or (tokenizer_root / "tokenizer_config.json").exists()):
        tokenizer_source = effective_base_model

    quant_summary = summarize_qalora_quantization(payload)
    metadata = {
        "format": "qalora_state",
        "num_modules": len(payload.get("qalora_state", {})),
        "checkpoint_label": payload.get("checkpoint_label"),
        "quant_types": quant_summary["quant_types"],
        "quant_bits": quant_summary["quant_bits"],
        "group_sizes": quant_summary["group_sizes"],
        "ranks": quant_summary["ranks"],
        "adapter_group_sizes": quant_summary["adapter_group_sizes"],
        "train_torch_dtype": payload.get("args", {}).get("torch_dtype"),
    }
    return effective_base_model, str(qalora_state_path), metadata, payload, tokenizer_source


def resolve_quzo_artifact(
    base_model_path: str,
    quzo_artifact_path: Optional[str],
) -> tuple[str, Optional[str], Optional[dict], Optional[dict], Optional[str]]:
    """
    Resolve the effective base model path and optional QuZO checkpoint path.

    Returns:
        effective_base_model_path, quzo_path, metadata, quzo_payload, tokenizer_source
    """
    artifact_path = quzo_artifact_path
    base_model_quzo_state = resolve_quzo_state_path(base_model_path)
    if artifact_path is None and base_model_quzo_state is not None:
        artifact_path = base_model_path

    if artifact_path is None:
        return base_model_path, None, None, None, base_model_path

    quzo_state_path = resolve_quzo_state_path(artifact_path)
    if quzo_state_path is None:
        raise ValueError(f"{artifact_path} does not point to a QuZO checkpoint directory or quzo_state.pt file.")

    payload = load_quzo_checkpoint(str(quzo_state_path))
    if base_model_quzo_state is not None:
        effective_base_model = payload.get("base_model_name_or_path")
    else:
        effective_base_model = base_model_path

    if not effective_base_model:
        raise ValueError(
            "Could not resolve the base model path for the QuZO checkpoint. "
            "Pass --base_model explicitly or ensure the checkpoint metadata contains it."
        )

    tokenizer_root = quzo_state_path.parent
    tokenizer_source = str(tokenizer_root)
    if not ((tokenizer_root / "tokenizer.json").exists() or (tokenizer_root / "tokenizer_config.json").exists()):
        tokenizer_source = effective_base_model

    quant_summary = summarize_quzo_quantization(payload)
    metadata = {
        "format": "quzo_state",
        "num_modules": len(payload.get("search_state", {})),
        "checkpoint_label": payload.get("checkpoint_label"),
        "quant_types": quant_summary["quant_types"],
        "quant_bits": quant_summary["quant_bits"],
        "group_sizes": quant_summary["group_sizes"],
        "train_torch_dtype": payload.get("args", {}).get("torch_dtype"),
    }
    return effective_base_model, str(quzo_state_path), metadata, payload, tokenizer_source


def resolve_qzo_artifact(
    base_model_path: str,
    qzo_artifact_path: Optional[str],
) -> tuple[str, Optional[str], Optional[dict], Optional[dict], Optional[str]]:
    """
    Resolve the effective base model path and optional QZO checkpoint path.

    Returns:
        effective_base_model_path, qzo_path, metadata, qzo_payload, tokenizer_source
    """
    artifact_path = qzo_artifact_path
    base_model_qzo_state = resolve_qzo_state_path(base_model_path)
    if artifact_path is None and base_model_qzo_state is not None:
        artifact_path = base_model_path

    if artifact_path is None:
        return base_model_path, None, None, None, base_model_path

    qzo_state_path = resolve_qzo_state_path(artifact_path)
    if qzo_state_path is None:
        raise ValueError(f"{artifact_path} does not point to a QZO checkpoint directory or qzo_state.pt file.")

    payload = load_qzo_checkpoint(str(qzo_state_path))
    if base_model_qzo_state is not None:
        effective_base_model = payload.get("base_model_name_or_path")
    else:
        effective_base_model = base_model_path

    if not effective_base_model:
        raise ValueError(
            "Could not resolve the base model path for the QZO checkpoint. "
            "Pass --base_model explicitly or ensure the checkpoint metadata contains it."
        )

    tokenizer_root = qzo_state_path.parent
    tokenizer_source = str(tokenizer_root)
    if not ((tokenizer_root / "tokenizer.json").exists() or (tokenizer_root / "tokenizer_config.json").exists()):
        tokenizer_source = effective_base_model

    quant_summary = summarize_qzo_quantization(payload)
    metadata = {
        "format": "qzo_state",
        "num_modules": len(payload.get("search_state", {})),
        "checkpoint_label": payload.get("checkpoint_label"),
        "quant_types": quant_summary["quant_types"],
        "quant_bits": quant_summary["quant_bits"],
        "group_sizes": quant_summary["group_sizes"],
        "train_torch_dtype": payload.get("args", {}).get("torch_dtype"),
    }
    return effective_base_model, str(qzo_state_path), metadata, payload, tokenizer_source


def resolve_qes_artifact(
    base_model_path: str,
    qes_artifact_path: Optional[str],
) -> tuple[str, Optional[str], Optional[dict], Optional[dict], Optional[str]]:
    """
    Resolve the effective base model path and optional QES checkpoint path.

    Returns:
        effective_base_model_path, qes_path, metadata, qes_payload, tokenizer_source
    """
    artifact_path = qes_artifact_path
    base_model_qes_state = resolve_qes_state_path(base_model_path)
    if artifact_path is None and base_model_qes_state is not None:
        artifact_path = base_model_path

    if artifact_path is None:
        return base_model_path, None, None, None, base_model_path

    qes_state_path = resolve_qes_state_path(artifact_path)
    if qes_state_path is None:
        raise ValueError(f"{artifact_path} does not point to a QES checkpoint directory or qes_state.pt file.")

    payload = load_qes_checkpoint(str(qes_state_path))
    if base_model_qes_state is not None:
        effective_base_model = payload.get("base_model_name_or_path")
    else:
        effective_base_model = base_model_path

    if not effective_base_model:
        raise ValueError(
            "Could not resolve the base model path for the QES checkpoint. "
            "Pass --base_model explicitly or ensure the checkpoint metadata contains it."
        )

    tokenizer_root = qes_state_path.parent
    tokenizer_source = str(tokenizer_root)
    if not ((tokenizer_root / "tokenizer.json").exists() or (tokenizer_root / "tokenizer_config.json").exists()):
        tokenizer_source = effective_base_model

    quant_summary = summarize_qes_quantization(payload)
    metadata = {
        "format": "qes_state",
        "num_modules": len(payload.get("search_state", {})),
        "checkpoint_label": payload.get("checkpoint_label"),
        "quant_types": quant_summary["quant_types"],
        "quant_bits": quant_summary["quant_bits"],
        "group_sizes": quant_summary["group_sizes"],
        "train_torch_dtype": payload.get("args", {}).get("torch_dtype"),
    }
    return effective_base_model, str(qes_state_path), metadata, payload, tokenizer_source


def get_bitsandbytes():
    """Import bitsandbytes lazily so non-4bit adapter paths keep working without it."""
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise ImportError(
            "adapter_bits=4 requires bitsandbytes to be installed in the runtime environment."
        ) from exc
    return bnb


def build_nf4_linear_from_linear(linear: nn.Linear, compute_dtype: torch.dtype) -> nn.Module:
    """
    Clone a dense LoRA linear layer into a bitsandbytes NF4 linear layer.

    bitsandbytes quantizes the weight when the module is moved to CUDA after the
    dense weights have been loaded, so we instantiate on CPU first, load the fp/bf16
    weights, and then move back to the original device.
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"Expected nn.Linear LoRA module, got {type(linear)}")

    bnb = get_bitsandbytes()
    quant_linear = bnb.nn.LinearNF4(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        compute_dtype=compute_dtype,
        compress_statistics=True,
    )

    state_dict = {
        "weight": linear.weight.detach().to(device="cpu", dtype=compute_dtype).contiguous(),
    }
    if linear.bias is not None:
        state_dict["bias"] = linear.bias.detach().to(device="cpu", dtype=compute_dtype).contiguous()

    quant_linear.load_state_dict(state_dict, strict=False)
    quant_linear = quant_linear.to(device=linear.weight.device)
    quant_linear.requires_grad_(False)
    return quant_linear


def resolve_parent_module_and_child_name(model: nn.Module, full_name: str) -> tuple[nn.Module, str]:
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def build_dense_linear_from_weight(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    target_dtype: torch.dtype,
    target_device: torch.device,
) -> nn.Linear:
    linear = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=bias is not None,
        device=target_device,
        dtype=target_dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(weight.to(device=target_device, dtype=target_dtype))
        if bias is not None:
            linear.bias.copy_(bias.to(device=target_device, dtype=target_dtype))
    linear.requires_grad_(False)
    return linear


def merge_loqt_adapters_for_eval(
    model: nn.Module,
    *,
    base_bits: int,
    is_main_process: bool = True,
) -> list[str]:
    if iter_loqt_modules is None:
        raise ImportError("LoQT merge-for-eval requires loqt.py to export iter_loqt_modules.")

    merged_modules = []
    loqt_modules = list(iter_loqt_modules(model))
    if not loqt_modules:
        return merged_modules

    for module_name, module in loqt_modules:
        merged_weight = module.materialize_weight(capture_grad=False, dtype=torch.float32)
        merged_bias = module.bias.detach().to(torch.float32) if module.bias is not None else None
        target_device = merged_weight.device

        dense_linear = build_dense_linear_from_weight(
            merged_weight,
            merged_bias,
            target_dtype=torch.float32,
            target_device=target_device,
        )

        if base_bits == 4:
            replacement = build_nf4_linear_from_linear(
                dense_linear,
                compute_dtype=get_compute_dtype(base_bits),
            )
        elif base_bits == 16:
            replacement = dense_linear.to(dtype=torch.bfloat16)
        else:
            replacement = dense_linear.to(dtype=torch.float32)

        replacement.requires_grad_(False)
        parent, child_name = resolve_parent_module_and_child_name(model, module_name)
        setattr(parent, child_name, replacement)
        merged_modules.append(module_name)

    if is_main_process:
        print("-" * 80)
        print("Merged LoQT adapter_b into static evaluation weights")
        print(f"Replaced LoQT modules: {len(merged_modules)}")
        print(f"Target precision after merge: {base_bits}-bit")
        for name in merged_modules[:8]:
            print(f"  {name}")
        if len(merged_modules) > 8:
            print(f"  ... and {len(merged_modules) - 8} more LoQT modules")

    return merged_modules


def _materialize_linear_weight(weight: torch.Tensor) -> torch.Tensor:
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        bnb = get_bitsandbytes()
        return bnb.functional.dequantize_4bit(weight.data, quant_state).detach()
    if hasattr(weight, "dequantize"):
        return weight.dequantize().detach()
    return weight.detach()


def _active_lora_adapter_names(module: nn.Module) -> list[str]:
    lora_a = getattr(module, "lora_A", None)
    lora_b = getattr(module, "lora_B", None)
    if not isinstance(lora_a, nn.ModuleDict) or not isinstance(lora_b, nn.ModuleDict):
        return []

    adapter_names = getattr(module, "active_adapters", None)
    if callable(adapter_names):
        adapter_names = adapter_names()
    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]
    if not adapter_names:
        adapter_names = sorted(set(lora_a.keys()) & set(lora_b.keys()))

    return [name for name in adapter_names if name in lora_a and name in lora_b]


def _compute_lora_delta_weight(module: nn.Module, adapter_name: str) -> torch.Tensor:
    lora_a = module.lora_A[adapter_name]
    lora_b = module.lora_B[adapter_name]

    weight_a = _materialize_linear_weight(lora_a.weight).to(torch.float32)
    weight_b = _materialize_linear_weight(lora_b.weight).to(torch.float32)
    delta = weight_b @ weight_a

    scaling = getattr(module, "scaling", {}).get(adapter_name, 1.0)
    delta = delta * float(scaling)

    if getattr(module, "fan_in_fan_out", False):
        delta = delta.T

    return delta


def merge_lora_adapters_into_static_eval_weights(
    peft_model: nn.Module,
    *,
    base_bits: int,
    is_main_process: bool = True,
) -> tuple[nn.Module, list[str]]:
    """
    Materialize PEFT LoRA adapters into the base model and remove LoRA wrappers.

    PEFT's generic merge path can be opaque for bitsandbytes-loaded backbones.
    For evaluation we explicitly form base_weight + LoRA_delta, then rebuild each
    touched layer in the requested base precision. With base_bits=4 this gives a
    static bitsandbytes NF4 layer containing the merged adapter update.
    """
    base_model = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    merged_modules: list[str] = []
    skipped_modules: list[str] = []

    for module_name, module in list(base_model.named_modules()):
        base_layer = getattr(module, "base_layer", None)
        adapter_names = _active_lora_adapter_names(module)
        if base_layer is None or not adapter_names:
            continue
        if getattr(module, "disable_adapters", False):
            skipped_modules.append(f"{module_name}: adapters disabled")
            continue
        if getattr(module, "use_dora", {}).get(adapter_names[0], False):
            skipped_modules.append(f"{module_name}: DoRA merge is not implemented")
            continue
        if not hasattr(base_layer, "weight"):
            skipped_modules.append(f"{module_name}: base layer has no weight")
            continue

        merged_weight = _materialize_linear_weight(base_layer.weight).to(torch.float32)
        for adapter_name in adapter_names:
            merged_weight = merged_weight + _compute_lora_delta_weight(module, adapter_name).to(
                device=merged_weight.device,
                dtype=torch.float32,
            )

        base_bias = getattr(base_layer, "bias", None)
        merged_bias = base_bias.detach().to(torch.float32) if base_bias is not None else None
        target_device = merged_weight.device

        if base_bits == 4:
            dense_linear = build_dense_linear_from_weight(
                merged_weight,
                merged_bias,
                target_dtype=torch.float32,
                target_device=target_device,
            )
            replacement = build_nf4_linear_from_linear(
                dense_linear,
                compute_dtype=get_compute_dtype(base_bits),
            )
        elif base_bits == 16:
            replacement = build_dense_linear_from_weight(
                merged_weight,
                merged_bias,
                target_dtype=torch.bfloat16,
                target_device=target_device,
            )
        else:
            replacement = build_dense_linear_from_weight(
                merged_weight,
                merged_bias,
                target_dtype=torch.float32,
                target_device=target_device,
            )

        replacement.requires_grad_(False)
        parent, child_name = resolve_parent_module_and_child_name(base_model, module_name)
        setattr(parent, child_name, replacement)
        merged_modules.append(module_name)

    if is_main_process:
        print("-" * 80)
        print("Merged LoRA adapters into static evaluation weights")
        print(f"Replaced LoRA-wrapped modules: {len(merged_modules)}")
        print(f"Target precision after merge: {base_bits}-bit")
        if base_bits == 4:
            print("Merged target modules were re-quantized as bitsandbytes NF4 Linear layers.")
        for name in merged_modules[:8]:
            print(f"  {name}")
        if len(merged_modules) > 8:
            print(f"  ... and {len(merged_modules) - 8} more LoRA modules")
        if skipped_modules:
            print("Skipped LoRA modules:")
            for name in skipped_modules[:8]:
                print(f"  {name}")
            if len(skipped_modules) > 8:
                print(f"  ... and {len(skipped_modules) - 8} more skipped modules")

    if not merged_modules:
        raise ValueError("merge_adapter was requested, but no active LoRA adapter modules were found to merge.")

    return base_model, merged_modules


def apply_nf4_to_lora_linears(model, compute_dtype: torch.dtype, is_main_process: bool = True):
    """Replace LoRA A/B linears with bitsandbytes NF4 linears for true 4-bit adapter loading."""
    bnb = get_bitsandbytes()

    converted = []
    skipped = []

    for module_name, module in list(model.named_modules()):
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if not isinstance(lora_a, nn.ModuleDict) or not isinstance(lora_b, nn.ModuleDict):
            continue

        adapter_names = sorted(set(lora_a.keys()) & set(lora_b.keys()))
        if not adapter_names:
            continue

        # The generic LoRA forward path casts inputs to lora_A.weight.dtype. That is
        # appropriate for dense weights but wrong for bitsandbytes Params4bit, whose
        # storage dtype is not the compute dtype. Keep the upstream activation dtype.
        if hasattr(module, "cast_input_dtype_enabled"):
            module.cast_input_dtype_enabled = False

        for adapter_name in adapter_names:
            old_a = lora_a[adapter_name]
            old_b = lora_b[adapter_name]

            if isinstance(old_a, bnb.nn.LinearNF4) and isinstance(old_b, bnb.nn.LinearNF4):
                continue

            if not isinstance(old_a, nn.Linear) or not isinstance(old_b, nn.Linear):
                skipped.append(f"{module_name}[{adapter_name}]")
                continue

            lora_a[adapter_name] = build_nf4_linear_from_linear(old_a, compute_dtype=compute_dtype)
            lora_b[adapter_name] = build_nf4_linear_from_linear(old_b, compute_dtype=compute_dtype)

            converted.append((f"{module_name}.lora_A.{adapter_name}", type(lora_a[adapter_name]).__name__))
            converted.append((f"{module_name}.lora_B.{adapter_name}", type(lora_b[adapter_name]).__name__))

    if is_main_process:
        print('-' * 80)
        print('Adapter precision summary')
        print('-' * 80)
        if not converted:
            print('No LoRA linear modules were converted to NF4.')
        else:
            print('Loaded LoRA A/B modules as bitsandbytes LinearNF4 layers.')
            print('Disabled LoRA input casting on converted layers to preserve the activation compute dtype.')
            print(f'Number of LoRA submodules converted: {len(converted)}')
            for name, module_type in converted[:8]:
                print(f'  {name}: module_type={module_type}')
            if len(converted) > 8:
                print(f'  ... and {len(converted) - 8} more LoRA submodules')
        if skipped:
            print('Skipped non-linear LoRA submodules:')
            for name in skipped[:8]:
                print(f'  {name}')
            if len(skipped) > 8:
                print(f'  ... and {len(skipped) - 8} more skipped submodules')


def apply_adapter_precision(model, adapter_bits: int, base_bits: int, is_main_process: bool = True):
    """Apply the requested precision policy to LoRA adapter parameters."""
    if adapter_bits == 4:
        apply_nf4_to_lora_linears(
            model,
            compute_dtype=get_compute_dtype(base_bits),
            is_main_process=is_main_process,
        )
        return

    lora_patterns = (
        'lora_A',
        'lora_B',
        'lora_embedding_A',
        'lora_embedding_B',
        'lora_magnitude_vector',
    )

    changed = []
    target_dtype = get_compute_dtype(32 if adapter_bits == 32 else 16)

    for name, param in model.named_parameters():
        if not any(pattern in name for pattern in lora_patterns):
            continue

        with torch.no_grad():
            if adapter_bits == 16:
                param.data = param.data.to(dtype=torch.bfloat16)
            else:  # 32-bit
                param.data = param.data.to(dtype=torch.float32)

        changed.append((name, tuple(param.shape), str(param.dtype)))

    if is_main_process:
        print('-' * 80)
        print('Adapter precision summary')
        print('-' * 80)
        if not changed:
            print('No LoRA parameters found to cast/quantize.')
        else:
            print(f'Cast LoRA parameters to {target_dtype}.')
            print(f'Number of LoRA parameters processed: {len(changed)}')
            for name, shape, dtype in changed[:8]:
                print(f'  {name}: shape={shape}, dtype={dtype}')
            if len(changed) > 8:
                print(f'  ... and {len(changed) - 8} more LoRA parameters')



def summarize_loaded_model(model, is_main_process: bool = True):
    """Print key precision/debug information after model loading."""
    if not is_main_process:
        return

    print('-' * 80)
    print('DEBUG: Model precision / module summary')
    print('-' * 80)

    loaded_in_4bit = getattr(model, 'is_loaded_in_4bit', None)
    if loaded_in_4bit is None and hasattr(model, 'base_model'):
        loaded_in_4bit = getattr(model.base_model, 'is_loaded_in_4bit', None)
    print(f'is_loaded_in_4bit: {loaded_in_4bit}')

    loaded_in_8bit = getattr(model, 'is_loaded_in_8bit', None)
    if loaded_in_8bit is None and hasattr(model, 'base_model'):
        loaded_in_8bit = getattr(model.base_model, 'is_loaded_in_8bit', None)
    print(f'is_loaded_in_8bit: {loaded_in_8bit}')

    for name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if weight is not None:
            print(f'First weighted module: {name}')
            print(f'  module type: {type(module)}')
            print(f'  weight dtype: {getattr(weight, "dtype", "N/A")}')
            break

    lora_found = False
    for name, param in model.named_parameters():
        if 'lora_' in name.lower():
            if not lora_found:
                print('Sample LoRA parameters:')
                lora_found = True
            print(f'  {name}: dtype={param.dtype}, shape={tuple(param.shape)}')
            break

    if not lora_found:
        print('No LoRA parameters detected in loaded model.')
        for name, module in model.named_modules():
            if type(module).__name__ == "LoQTLinear":
                print(f'Sample LoQT module: {name} ({type(module)})')
                break
            if type(module).__name__ == "QALoRALinear":
                print(f'Sample QA-LoRA module: {name} ({type(module)})')
                break


def load_model_with_peft(
    base_model_path: str,
    lora_model_path: Optional[str] = None,
    gradcodes_artifact_path: Optional[str] = None,
    pvtuning_artifact_path: Optional[str] = None,
    loqt_artifact_path: Optional[str] = None,
    qalora_artifact_path: Optional[str] = None,
    qzo_artifact_path: Optional[str] = None,
    quzo_artifact_path: Optional[str] = None,
    qes_artifact_path: Optional[str] = None,
    base_bits: Literal[4, 16, 32] = 4,
    adapter_bits: Literal[4, 16, 32] = 16,
    device_map: str = "auto",
    merge_adapter: bool = False,
    merge_loqt_adapter: bool = True,
    distributed_state=None,
):
    """
    Load model with optional quantization and optional LoRA adapters

    Args:
        base_model_path: Path to base model
        lora_model_path: Path to LoRA adapters (optional)
        gradcodes_artifact_path: Path to a Gradcodes discrete artifact directory (optional)
        pvtuning_artifact_path: Path to a PV-Tuning checkpoint directory/file (optional)
        loqt_artifact_path: Path to a LoQT checkpoint directory/file (optional)
        qalora_artifact_path: Path to a QA-LoRA checkpoint directory/file (optional)
        qzo_artifact_path: Path to a QZO checkpoint directory/file (optional)
        quzo_artifact_path: Path to a QuZO checkpoint directory/file (optional)
        qes_artifact_path: Path to a QES checkpoint directory/file (optional)
        base_bits: Base model precision (4=NF4, 16=BF16/FP16, 32=float32)
        adapter_bits: LoRA adapter precision (4, 16, or 32)
        device_map: Device mapping strategy

    Returns:
        tuple: (model, tokenizer)
    """
    # If distributed_state not provided, check it
    if distributed_state is None:
        distributed_state = setup_distributed()

    is_main_process = True
    if distributed_state is not None:
        is_main_process = distributed_state.is_main_process
        # In distributed mode, use the device assigned to this process
        device_map = {"": distributed_state.device}

    specified_artifact_count = sum(
        path is not None
        for path in (
            gradcodes_artifact_path,
            pvtuning_artifact_path,
            loqt_artifact_path,
            qalora_artifact_path,
            qzo_artifact_path,
            quzo_artifact_path,
            qes_artifact_path,
        )
    )
    if specified_artifact_count > 1:
        raise ValueError(
            "Specify at most one of --gradcodes_artifact, --pvtuning_artifact, --loqt_artifact, "
            "--qalora_artifact, --qzo_artifact, --quzo_artifact, and --qes_artifact."
        )

    resolved_artifact_path = None
    artifact_metadata = None
    search_checkpoint_payload = None
    loqt_checkpoint_payload = None
    qalora_checkpoint_payload = None
    qzo_checkpoint_payload = None
    quzo_checkpoint_payload = None
    qes_checkpoint_payload = None
    tokenizer_source = base_model_path
    effective_base_model_path = base_model_path
    artifact_label = "Gradcodes input"
    base_model_search_state_path = resolve_gradcodes_search_state_path(base_model_path)
    base_model_search_payload = load_gradcodes_search_checkpoint(str(base_model_search_state_path)) if base_model_search_state_path is not None else None

    auto_detect_qalora = (
        qalora_artifact_path is None
        and loqt_artifact_path is None
        and gradcodes_artifact_path is None
        and pvtuning_artifact_path is None
        and qzo_artifact_path is None
        and quzo_artifact_path is None
        and qes_artifact_path is None
        and resolve_qalora_state_path(base_model_path) is not None
    )
    auto_detect_loqt = (
        loqt_artifact_path is None
        and qalora_artifact_path is None
        and gradcodes_artifact_path is None
        and pvtuning_artifact_path is None
        and qzo_artifact_path is None
        and quzo_artifact_path is None
        and qes_artifact_path is None
        and resolve_loqt_state_path(base_model_path) is not None
    )
    auto_detect_qzo = (
        qzo_artifact_path is None
        and qalora_artifact_path is None
        and loqt_artifact_path is None
        and gradcodes_artifact_path is None
        and pvtuning_artifact_path is None
        and quzo_artifact_path is None
        and qes_artifact_path is None
        and resolve_qzo_state_path(base_model_path) is not None
    )
    auto_detect_quzo = (
        quzo_artifact_path is None
        and qalora_artifact_path is None
        and loqt_artifact_path is None
        and gradcodes_artifact_path is None
        and pvtuning_artifact_path is None
        and qzo_artifact_path is None
        and qes_artifact_path is None
        and resolve_quzo_state_path(base_model_path) is not None
    )
    auto_detect_qes = (
        qes_artifact_path is None
        and qalora_artifact_path is None
        and loqt_artifact_path is None
        and gradcodes_artifact_path is None
        and pvtuning_artifact_path is None
        and qzo_artifact_path is None
        and quzo_artifact_path is None
        and resolve_qes_state_path(base_model_path) is not None
    )
    auto_detect_pvtuning = (
        pvtuning_artifact_path is None
        and qalora_artifact_path is None
        and loqt_artifact_path is None
        and gradcodes_artifact_path is None
        and qzo_artifact_path is None
        and quzo_artifact_path is None
        and qes_artifact_path is None
        and base_model_search_payload is not None
        and is_pvtuning_search_checkpoint_payload(base_model_search_payload)
    )
    if qalora_artifact_path is not None or auto_detect_qalora:
        effective_base_model_path, resolved_artifact_path, artifact_metadata, qalora_checkpoint_payload, tokenizer_source = resolve_qalora_artifact(
            base_model_path=base_model_path,
            qalora_artifact_path=qalora_artifact_path,
        )
        artifact_label = "QA-LoRA input"
    elif loqt_artifact_path is not None or auto_detect_loqt:
        effective_base_model_path, resolved_artifact_path, artifact_metadata, loqt_checkpoint_payload, tokenizer_source = resolve_loqt_artifact(
            base_model_path=base_model_path,
            loqt_artifact_path=loqt_artifact_path,
        )
        artifact_label = "LoQT input"
    elif qzo_artifact_path is not None or auto_detect_qzo:
        effective_base_model_path, resolved_artifact_path, artifact_metadata, qzo_checkpoint_payload, tokenizer_source = resolve_qzo_artifact(
            base_model_path=base_model_path,
            qzo_artifact_path=qzo_artifact_path,
        )
        artifact_label = "QZO input"
    elif quzo_artifact_path is not None or auto_detect_quzo:
        effective_base_model_path, resolved_artifact_path, artifact_metadata, quzo_checkpoint_payload, tokenizer_source = resolve_quzo_artifact(
            base_model_path=base_model_path,
            quzo_artifact_path=quzo_artifact_path,
        )
        artifact_label = "QuZO input"
    elif qes_artifact_path is not None or auto_detect_qes:
        effective_base_model_path, resolved_artifact_path, artifact_metadata, qes_checkpoint_payload, tokenizer_source = resolve_qes_artifact(
            base_model_path=base_model_path,
            qes_artifact_path=qes_artifact_path,
        )
        artifact_label = "QES input"
    elif pvtuning_artifact_path is not None or auto_detect_pvtuning:
        effective_base_model_path, resolved_artifact_path, artifact_metadata, search_checkpoint_payload, tokenizer_source = resolve_pvtuning_artifact(
            base_model_path=base_model_path,
            pvtuning_artifact_path=pvtuning_artifact_path,
        )
        artifact_label = "PV-Tuning input"
    else:
        effective_base_model_path, resolved_artifact_path, artifact_metadata, search_checkpoint_payload, tokenizer_source = resolve_gradcodes_artifact(
            base_model_path=base_model_path,
            gradcodes_artifact_path=gradcodes_artifact_path,
        )

    if is_main_process:
        print("=" * 80)
        print("Loading model for QLoRA evaluation...")
        print("=" * 80)
        print(f"Base model argument: {base_model_path}")
        print(f"Resolved base model path: {effective_base_model_path}")
        print(f"LoRA adapter: {lora_model_path if lora_model_path else 'None (base model only)'}")
        print(f"{artifact_label}: {resolved_artifact_path if resolved_artifact_path else 'None'}")
        print(f"Base model precision: {base_bits}-bit")
        print(f"Adapter precision: {adapter_bits}-bit")
        print(f"Device map: {device_map}")
        if artifact_metadata:
            print(f"Artifact modules: {artifact_metadata.get('num_modules', 'N/A')}")
            print(f"Artifact format: {artifact_metadata.get('format', 'N/A')}")
            if artifact_metadata.get("format") in {"gradcodes_search_state", "pvtuning_ds_search_state"}:
                print(f"Search-state label: {artifact_metadata.get('state_label', 'Gradcodes')}")
                print(f"Saved quant_type(s): {artifact_metadata.get('quant_types')}")
                print(f"Saved quant_bits: {artifact_metadata.get('quant_bits')}")
                print(f"Saved group_size(s): {artifact_metadata.get('group_sizes')}")
                print(f"Saved train torch_dtype: {artifact_metadata.get('train_torch_dtype')}")
                if artifact_metadata.get("format") == "pvtuning_ds_search_state" and base_bits == 4:
                    print(
                        "Note: PV-Tuning modules are reconstructed from the saved 4-bit "
                        "Gradcodes-compatible quantized search state."
                    )
            elif artifact_metadata.get("format") == "loqt_state":
                print(f"Saved quant_type(s): {artifact_metadata.get('quant_types')}")
                print(f"Saved quant_bits: {artifact_metadata.get('quant_bits')}")
                print(f"Saved group_size(s): {artifact_metadata.get('group_sizes')}")
                print(f"Saved rank(s): {artifact_metadata.get('ranks')}")
                print(f"Saved train torch_dtype: {artifact_metadata.get('train_torch_dtype')}")
                if base_bits == 4:
                    print(
                        "Note: untouched base layers stay bitsandbytes NF4, while LoQT target layers "
                        "are reconstructed from the saved LoQT quantized state."
                    )
            elif artifact_metadata.get("format") == "qalora_state":
                print(f"Saved quant_type(s): {artifact_metadata.get('quant_types')}")
                print(f"Saved quant_bits: {artifact_metadata.get('quant_bits')}")
                print(f"Saved group_size(s): {artifact_metadata.get('group_sizes')}")
                print(f"Saved adapter group_size(s): {artifact_metadata.get('adapter_group_sizes')}")
                print(f"Saved rank(s): {artifact_metadata.get('ranks')}")
                print(f"Saved train torch_dtype: {artifact_metadata.get('train_torch_dtype')}")
            elif artifact_metadata.get("format") == "qzo_state":
                print(f"Saved quant_type(s): {artifact_metadata.get('quant_types')}")
                print(f"Saved quant_bits: {artifact_metadata.get('quant_bits')}")
                print(f"Saved group_size(s): {artifact_metadata.get('group_sizes')}")
                print(f"Saved train torch_dtype: {artifact_metadata.get('train_torch_dtype')}")
                if base_bits == 4:
                    print(
                        "Note: QZO target layers are reconstructed from the saved 4-bit fixed-code "
                        "state and learned scale values; the exact checkpoint path loads the base "
                        "in the checkpoint training dtype instead of re-quantizing untouched layers "
                        "with bitsandbytes."
                    )
            elif artifact_metadata.get("format") == "quzo_state":
                print(f"Saved quant_type(s): {artifact_metadata.get('quant_types')}")
                print(f"Saved quant_bits: {artifact_metadata.get('quant_bits')}")
                print(f"Saved group_size(s): {artifact_metadata.get('group_sizes')}")
                print(f"Saved train torch_dtype: {artifact_metadata.get('train_torch_dtype')}")
                if base_bits == 4:
                    print(
                        "Note: QuZO target layers are reconstructed from the saved 4-bit code state; "
                        "the exact checkpoint path loads the base in the checkpoint training dtype "
                        "instead of re-quantizing untouched layers with bitsandbytes."
                    )
            elif artifact_metadata.get("format") == "qes_state":
                print(f"Saved quant_type(s): {artifact_metadata.get('quant_types')}")
                print(f"Saved quant_bits: {artifact_metadata.get('quant_bits')}")
                print(f"Saved group_size(s): {artifact_metadata.get('group_sizes')}")
                print(f"Saved train torch_dtype: {artifact_metadata.get('train_torch_dtype')}")
                if base_bits == 4:
                    print(
                        "Note: QES target layers are reconstructed from the saved 4-bit integer code state; "
                        "the exact checkpoint path loads the base in the checkpoint training dtype "
                        "instead of re-quantizing untouched layers with bitsandbytes."
                    )
            elif base_bits == 4:
                print(
                    "Note: Gradcodes-adapted modules are materialized as dense compute layers "
                    "on top of a 4-bit-loaded base model."
                )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    search_state_args = search_checkpoint_payload.get("args", {}) if search_checkpoint_payload is not None else {}
    qzo_state_args = qzo_checkpoint_payload.get("args", {}) if qzo_checkpoint_payload is not None else {}
    quzo_state_args = quzo_checkpoint_payload.get("args", {}) if quzo_checkpoint_payload is not None else {}
    qes_state_args = qes_checkpoint_payload.get("args", {}) if qes_checkpoint_payload is not None else {}
    exact_search_state_eval = search_checkpoint_payload is not None
    exact_qzo_state_eval = qzo_checkpoint_payload is not None
    exact_quzo_state_eval = quzo_checkpoint_payload is not None
    exact_qes_state_eval = qes_checkpoint_payload is not None

    # Configure base model precision
    if exact_search_state_eval or exact_qzo_state_eval or exact_quzo_state_eval or exact_qes_state_eval:
        if exact_search_state_eval:
            exact_state_args = search_state_args
            exact_label = artifact_metadata.get("state_label", "Gradcodes") if artifact_metadata else "Gradcodes"
        elif exact_qzo_state_eval:
            exact_state_args = qzo_state_args
            exact_label = "QZO"
        elif exact_quzo_state_eval:
            exact_state_args = quzo_state_args
            exact_label = "QuZO"
        else:
            exact_state_args = qes_state_args
            exact_label = "QES"
        train_torch_dtype = torch_dtype_from_name(exact_state_args.get("torch_dtype", "bfloat16"))
        if is_main_process:
            print(
                f"Base model: Loading in the checkpoint's training dtype for exact {exact_label} alignment "
                f"({exact_state_args.get('torch_dtype', 'bfloat16')})"
            )
            if base_bits == 4:
                print(
                    "  - CLI --base_bits=4 is not used to re-quantize untouched base layers here."
                )
                print(
                    f"  - Saved {exact_label} target modules are reconstructed from the checkpoint's "
                    "own quantization metadata (quant_type / quant_bits / group_size)."
                )
        model_kwargs = {
            "dtype": train_torch_dtype,
            "device_map": device_map,
        }
    elif base_bits == 4:
        if is_main_process:
            print(f"Base model: NF4 quantization enabled")
            print(f"  - Compute dtype: bfloat16")
            print(f"  - Double quantization: True")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": device_map,
        }
    elif base_bits == 16:
        if is_main_process:
            print("Base model: Loading in 16-bit (BF16)")
        model_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": device_map,
        }
    else:  # 32-bit
        if is_main_process:
            print("Base model: Loading in 32-bit (float32)")
        model_kwargs = {
            "dtype": torch.float32,
            "device_map": device_map,
        }

    # Load base model
    if is_main_process:
        print("-" * 80)
        print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        effective_base_model_path,
        trust_remote_code=True,
        **model_kwargs,
    )

    if search_checkpoint_payload is not None:
        if apply_search_state_to_model is None:
            raise ImportError(
                "Gradcodes/PV-Tuning search-state evaluation requires gradcodes.py to be importable."
            )
        replaced = apply_search_state_to_model(
            model,
            search_state=search_checkpoint_payload["search_state"],
            compute_dtype=model.dtype if hasattr(model, "dtype") else get_compute_dtype(16),
        )
        if is_main_process:
            print("-" * 80)
            state_label = artifact_metadata.get("state_label", "Gradcodes") if artifact_metadata else "Gradcodes"
            print(f"Applied {state_label} search-state checkpoint")
            print(f"Replaced modules: {replaced}")
    elif qzo_checkpoint_payload is not None:
        if apply_qzo_state_to_model is None:
            raise ImportError(
                "QZO checkpoint evaluation requires qzo.py to be importable."
            )
        replaced = apply_qzo_state_to_model(
            model,
            search_state=qzo_checkpoint_payload["search_state"],
            compute_dtype=model.dtype if hasattr(model, "dtype") else get_compute_dtype(16),
        )
        if is_main_process:
            print("-" * 80)
            print("Applied QZO checkpoint")
            print(f"Replaced modules: {replaced}")
    elif quzo_checkpoint_payload is not None:
        if apply_quzo_state_to_model is None:
            raise ImportError(
                "QuZO checkpoint evaluation requires quzo.py to be importable."
            )
        replaced = apply_quzo_state_to_model(
            model,
            search_state=quzo_checkpoint_payload["search_state"],
            compute_dtype=model.dtype if hasattr(model, "dtype") else get_compute_dtype(16),
        )
        if is_main_process:
            print("-" * 80)
            print("Applied QuZO checkpoint")
            print(f"Replaced modules: {replaced}")
    elif qes_checkpoint_payload is not None:
        if apply_qes_state_to_model is None:
            raise ImportError(
                "QES checkpoint evaluation requires qes.py to be importable."
            )
        replaced = apply_qes_state_to_model(
            model,
            search_state=qes_checkpoint_payload["search_state"],
            compute_dtype=model.dtype if hasattr(model, "dtype") else get_compute_dtype(16),
        )
        if is_main_process:
            print("-" * 80)
            print("Applied QES checkpoint")
            print(f"Replaced modules: {replaced}")
    elif loqt_checkpoint_payload is not None:
        if apply_loqt_state_to_model is None:
            raise ImportError(
                "LoQT checkpoint evaluation requires loqt.py to be importable."
            )
        replaced = apply_loqt_state_to_model(
            model,
            state=loqt_checkpoint_payload["loqt_state"],
        )
        if is_main_process:
            print("-" * 80)
            print("Applied LoQT checkpoint")
            print(f"Inserted/Replaced modules: {replaced if replaced else 'already wrapped'}")
        if merge_loqt_adapter:
            merge_loqt_adapters_for_eval(
                model,
                base_bits=base_bits,
                is_main_process=is_main_process,
            )
    elif qalora_checkpoint_payload is not None:
        if apply_qalora_state_to_model is None:
            raise ImportError(
                "QA-LoRA checkpoint evaluation requires qalora.py to be importable."
            )
        replaced = apply_qalora_state_to_model(
            model,
            state=qalora_checkpoint_payload["qalora_state"],
        )
        if is_main_process:
            print("-" * 80)
            print("Applied QA-LoRA checkpoint")
            print(f"Inserted/Replaced modules: {replaced if replaced else 'already wrapped'}")
    elif resolved_artifact_path is not None:
        _, artifact_state = load_gradcodes_artifact(resolved_artifact_path)
        replaced = apply_gradcodes_artifact_to_model(
            model,
            artifact_state=artifact_state,
            dtype=get_compute_dtype(base_bits),
        )
        if is_main_process:
            print("-" * 80)
            print("Applied Gradcodes artifact")
            print(f"Replaced modules: {replaced}")
   
    # Load LoRA adapters if provided
    if lora_model_path:
        if is_main_process:
            print(f"Loading LoRA adapters from {lora_model_path} in {adapter_bits}-bit...")

        # Load the adapter
        peft_model = PeftModel.from_pretrained(
            model,
            lora_model_path,
            is_trainable=False,
        )

        # Make adapter_bits actually affect the loaded LoRA weights
        apply_adapter_precision(
            peft_model,
            adapter_bits=adapter_bits,
            base_bits=base_bits,
            is_main_process=is_main_process,
        )

        # Optionally merge adapter into base model
        if merge_adapter:
            if is_main_process:
                print("Merging LoRA adapters into static base-model weights...")
            model, _ = merge_lora_adapters_into_static_eval_weights(
                peft_model,
                base_bits=base_bits,
                is_main_process=is_main_process,
            )
            if is_main_process:
                print("LoRA adapters merged successfully!")
        else:
            model = peft_model
            if is_main_process:
                print("LoRA adapters loaded (not merged, will be applied dynamically)")

    model.eval()

    summarize_loaded_model(model, is_main_process=is_main_process)

    if is_main_process:
        print("=" * 80)
        print("Model loaded successfully!")
        print("=" * 80)

    return model, tokenizer


def print_results(results: dict):
    """Print evaluation results in a formatted table"""
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    # Print results for each task
    if "results" in results:
        for task_name, task_results in results["results"].items():
            if isinstance(task_results, dict):
                print(f"\n{task_name}:")
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"  {metric_name}: {metric_value:.8f}")
                    else:
                        print(f"  {metric_name}: {metric_value}")

    # Print grouped results if available
    if "grouped" in results and results["grouped"]:
        print("\n" + "-" * 80)
        print("Grouped Results:")
        for group_name, group_result in results["grouped"].items():
            print(f"\n{group_name}:")
            if isinstance(group_result, dict):
                for metric_name, metric_value in group_result.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"  {metric_name}: {metric_value:.8f}")

    print("\n" + "=" * 80)


def run_evaluation(
    base_model: str,
    lora_model: Optional[str] = None,
    gradcodes_artifact: Optional[str] = None,
    pvtuning_artifact: Optional[str] = None,
    loqt_artifact: Optional[str] = None,
    qalora_artifact: Optional[str] = None,
    qzo_artifact: Optional[str] = None,
    quzo_artifact: Optional[str] = None,
    qes_artifact: Optional[str] = None,
    tasks: str = "mmlu",
    batch_size: int = 4,
    num_fewshot: int = 5,
    output_path: str = "eval_results.json",
    base_bits: int = 4,
    adapter_bits: int = 16,
    merge_adapter: bool = False,
    merge_loqt_adapter: bool = True,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
) -> dict:
    """
    Run lm-eval-harness evaluation with QLoRA model

    Args:
        base_model: Base model path or HF model ID
        lora_model: Path to LoRA adapter directory (optional)
        gradcodes_artifact: Path to a Gradcodes discrete artifact directory (optional)
        pvtuning_artifact: Path to a PV-Tuning checkpoint directory/file (optional)
        loqt_artifact: Path to a LoQT checkpoint directory/file (optional)
        qalora_artifact: Path to a QA-LoRA checkpoint directory/file (optional)
        qzo_artifact: Path to a QZO checkpoint directory/file (optional)
        quzo_artifact: Path to a QuZO checkpoint directory/file (optional)
        qes_artifact: Path to a QES checkpoint directory/file (optional)
        tasks: Tasks to evaluate (comma-separated for multiple)
        batch_size: Batch size for evaluation
        num_fewshot: Number of few-shot examples
        output_path: Path to save results
        base_bits: Base model precision (4, 16, or 32)
        adapter_bits: LoRA adapter precision (4, 16, or 32)
        apply_chat_template: Whether to format prompts with the tokenizer chat template
        fewshot_as_multiturn: Whether few-shot examples should be represented as multi-turn chat history

    Returns:
        Evaluation results dictionary
    """
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model

    distributed_state = setup_distributed()
    is_main_process = True
    if distributed_state is not None:
        is_main_process = distributed_state.is_main_process

    specified_artifact_count = sum(
        path is not None
        for path in (
            gradcodes_artifact,
            pvtuning_artifact,
            loqt_artifact,
            qalora_artifact,
            qzo_artifact,
            quzo_artifact,
            qes_artifact,
        )
    )
    if specified_artifact_count > 1:
        raise ValueError(
            "Specify at most one of --gradcodes_artifact, --pvtuning_artifact, --loqt_artifact, "
            "--qalora_artifact, --qzo_artifact, --quzo_artifact, and --qes_artifact."
        )

    artifact_label = "Gradcodes input"
    base_model_search_state_path = resolve_gradcodes_search_state_path(base_model)
    base_model_search_payload = load_gradcodes_search_checkpoint(str(base_model_search_state_path)) if base_model_search_state_path is not None else None
    auto_detect_qalora = (
        qalora_artifact is None
        and loqt_artifact is None
        and gradcodes_artifact is None
        and pvtuning_artifact is None
        and qzo_artifact is None
        and quzo_artifact is None
        and qes_artifact is None
        and resolve_qalora_state_path(base_model) is not None
    )
    auto_detect_loqt = (
        loqt_artifact is None
        and qalora_artifact is None
        and gradcodes_artifact is None
        and pvtuning_artifact is None
        and qzo_artifact is None
        and quzo_artifact is None
        and qes_artifact is None
        and resolve_loqt_state_path(base_model) is not None
    )
    auto_detect_qzo = (
        qzo_artifact is None
        and qalora_artifact is None
        and loqt_artifact is None
        and gradcodes_artifact is None
        and pvtuning_artifact is None
        and quzo_artifact is None
        and qes_artifact is None
        and resolve_qzo_state_path(base_model) is not None
    )
    auto_detect_quzo = (
        quzo_artifact is None
        and qalora_artifact is None
        and loqt_artifact is None
        and gradcodes_artifact is None
        and pvtuning_artifact is None
        and qzo_artifact is None
        and qes_artifact is None
        and resolve_quzo_state_path(base_model) is not None
    )
    auto_detect_qes = (
        qes_artifact is None
        and qalora_artifact is None
        and loqt_artifact is None
        and gradcodes_artifact is None
        and pvtuning_artifact is None
        and qzo_artifact is None
        and quzo_artifact is None
        and resolve_qes_state_path(base_model) is not None
    )
    auto_detect_pvtuning = (
        pvtuning_artifact is None
        and qalora_artifact is None
        and loqt_artifact is None
        and gradcodes_artifact is None
        and qzo_artifact is None
        and quzo_artifact is None
        and qes_artifact is None
        and base_model_search_payload is not None
        and is_pvtuning_search_checkpoint_payload(base_model_search_payload)
    )
    if qalora_artifact is not None or auto_detect_qalora:
        effective_base_model, resolved_artifact_path, artifact_metadata, _, tokenizer_source = resolve_qalora_artifact(
            base_model_path=base_model,
            qalora_artifact_path=qalora_artifact,
        )
        artifact_label = "QA-LoRA input"
    elif loqt_artifact is not None or auto_detect_loqt:
        effective_base_model, resolved_artifact_path, artifact_metadata, _, tokenizer_source = resolve_loqt_artifact(
            base_model_path=base_model,
            loqt_artifact_path=loqt_artifact,
        )
        artifact_label = "LoQT input"
    elif qzo_artifact is not None or auto_detect_qzo:
        effective_base_model, resolved_artifact_path, artifact_metadata, _, tokenizer_source = resolve_qzo_artifact(
            base_model_path=base_model,
            qzo_artifact_path=qzo_artifact,
        )
        artifact_label = "QZO input"
    elif quzo_artifact is not None or auto_detect_quzo:
        effective_base_model, resolved_artifact_path, artifact_metadata, _, tokenizer_source = resolve_quzo_artifact(
            base_model_path=base_model,
            quzo_artifact_path=quzo_artifact,
        )
        artifact_label = "QuZO input"
    elif qes_artifact is not None or auto_detect_qes:
        effective_base_model, resolved_artifact_path, artifact_metadata, _, tokenizer_source = resolve_qes_artifact(
            base_model_path=base_model,
            qes_artifact_path=qes_artifact,
        )
        artifact_label = "QES input"
    elif pvtuning_artifact is not None or auto_detect_pvtuning:
        effective_base_model, resolved_artifact_path, artifact_metadata, _, tokenizer_source = resolve_pvtuning_artifact(
            base_model_path=base_model,
            pvtuning_artifact_path=pvtuning_artifact,
        )
        artifact_label = "PV-Tuning input"
    else:
        effective_base_model, resolved_artifact_path, artifact_metadata, _, tokenizer_source = resolve_gradcodes_artifact(
            base_model_path=base_model,
            gradcodes_artifact_path=gradcodes_artifact,
        )

    if is_main_process:
        print("=" * 80)
        print("Running lm-eval-harness evaluation with QLoRA model")
        print("=" * 80)
        print(f"Base model argument: {base_model}")
        print(f"Resolved base model path: {effective_base_model}")
        print(f"LoRA adapter: {lora_model if lora_model else 'None (base model only)'}")
        print(f"{artifact_label}: {resolved_artifact_path if resolved_artifact_path else 'None'}")
        print(f"Base model precision: {base_bits}-bit")
        print(f"Adapter precision: {adapter_bits}-bit")
        print(f"Tasks: {tasks}")
        print(f"Batch size: {batch_size}")
        print(f"Few-shot examples: {num_fewshot}")
        if artifact_metadata:
            print(f"Artifact modules: {artifact_metadata.get('num_modules', 'N/A')}")
        print("=" * 80)

    # Create custom HFLM class that loads our QLoRA model
    class QLoRAHFLM(HFLM):
        def __init__(self, pretrained: str, **kwargs):
            self._pretrained = pretrained
            self._lora_path = lora_model
            self._gradcodes_artifact_path = resolved_artifact_path if artifact_label == "Gradcodes input" else None
            self._pvtuning_artifact_path = resolved_artifact_path if artifact_label == "PV-Tuning input" else None
            self._loqt_artifact_path = resolved_artifact_path if artifact_label == "LoQT input" else None
            self._qalora_artifact_path = resolved_artifact_path if artifact_label == "QA-LoRA input" else None
            self._qzo_artifact_path = resolved_artifact_path if artifact_label == "QZO input" else None
            self._quzo_artifact_path = resolved_artifact_path if artifact_label == "QuZO input" else None
            self._qes_artifact_path = resolved_artifact_path if artifact_label == "QES input" else None
            self._base_bits = base_bits
            self._adapter_bits = adapter_bits
            self._merge_adapter = merge_adapter
            self._merge_loqt_adapter = merge_loqt_adapter
            super().__init__(pretrained=pretrained, **kwargs)

        def _create_model(self, pretrained: str, **kwargs):
            model, tokenizer = load_model_with_peft(
                base_model_path=pretrained,
                lora_model_path=self._lora_path,
                gradcodes_artifact_path=self._gradcodes_artifact_path,
                pvtuning_artifact_path=self._pvtuning_artifact_path,
                loqt_artifact_path=self._loqt_artifact_path,
                qalora_artifact_path=self._qalora_artifact_path,
                qzo_artifact_path=self._qzo_artifact_path,
                quzo_artifact_path=self._quzo_artifact_path,
                qes_artifact_path=self._qes_artifact_path,
                base_bits=self._base_bits,
                adapter_bits=self._adapter_bits,
                device_map="auto",
                merge_adapter=self._merge_adapter,
                merge_loqt_adapter=self._merge_loqt_adapter,
                distributed_state=distributed_state,
            )

            self._model = model
            # HFLM usually creates the tokenizer first; only override it when explicitly requested.
            self.tokenizer = tokenizer
            if is_main_process:
                print("-" * 80)
                print("DEBUG: Model Information")
                print("-" * 80)
                print("is_loaded_in_4bit:", getattr(model, "is_loaded_in_4bit", False))
                if self._lora_path==None or self._merge_adapter:
                    m = model.model.layers[0].self_attn.q_proj
                else:
                    m = model.base_model.model.model.layers[0].self_attn.q_proj
                print(f"LoRA Layer type: {type(m)}")

                # Base layer parameters.
                if hasattr(m, "base_layer"):
                    print(f"Base layer type: {type(m.base_layer)}")
                    if hasattr(m.base_layer, "weight"):
                        print(f"Base layer weight dtype: {m.base_layer.weight.dtype}")

                # LoRA parameters
                if hasattr(m, "lora_A"):
                    print(f"LoRA A dtype: {m.lora_A['default'].weight.dtype}")
                if hasattr(m, "lora_B"):
                    print(f"LoRA B dtype: {m.lora_B['default'].weight.dtype}")

    # Register our custom model
    register_model("qlora_hf")(QLoRAHFLM)

    # Parse model args
    model_args = {
        "pretrained": effective_base_model,
    }
    if tokenizer_source != effective_base_model:
        model_args["tokenizer"] = tokenizer_source

    # Parse tasks
    task_list = tasks.split(",") if "," in tasks else [tasks]

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run evaluation
    results = evaluator.simple_evaluate(
        model="qlora_hf",
        model_args=model_args,
        tasks=task_list,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        device=device,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
    )

    # Print and save results (only on main process)
    if is_main_process:
        print_results(results)

    
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate QLoRA/base model with lm-eval-harness (supports multi-GPU via accelerate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base model path or HF model ID. You may also pass a compatible Gradcodes/PV-Tuning/LoQT/QA-LoRA/QZO/QuZO/QES checkpoint directory here and let the script auto-detect it.",
    )
    parser.add_argument(
        "--lora_model",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (optional, if not specified evaluates base model only)",
    )
    parser.add_argument(
        "--gradcodes_artifact",
        type=str,
        default=None,
        help="Path to either a legacy Gradcodes artifact directory or a Gradcodes search-state checkpoint directory/file. If omitted, compatible paths passed via --base_model are auto-detected.",
    )
    parser.add_argument(
        "--pvtuning_artifact",
        type=str,
        default=None,
        help="Path to a PV-Tuning checkpoint directory or gradcodes_state.pt file. If omitted, compatible paths passed via --base_model are auto-detected.",
    )
    parser.add_argument(
        "--loqt_artifact",
        type=str,
        default=None,
        help="Path to a LoQT checkpoint directory or loqt_state.pt file. If omitted, compatible paths passed via --base_model are auto-detected.",
    )
    parser.add_argument(
        "--qalora_artifact",
        type=str,
        default=None,
        help="Path to a QA-LoRA checkpoint directory or qalora_state.pt file. If omitted, compatible paths passed via --base_model are auto-detected.",
    )
    parser.add_argument(
        "--qzo_artifact",
        type=str,
        default=None,
        help="Path to a QZO checkpoint directory or qzo_state.pt file. If omitted, compatible paths passed via --base_model are auto-detected.",
    )
    parser.add_argument(
        "--quzo_artifact",
        type=str,
        default=None,
        help="Path to a QuZO checkpoint directory or quzo_state.pt file. If omitted, compatible paths passed via --base_model are auto-detected.",
    )
    parser.add_argument(
        "--qes_artifact",
        type=str,
        default=None,
        help="Path to a QES checkpoint directory or qes_state.pt file. If omitted, compatible paths passed via --base_model are auto-detected.",
    )
    parser.add_argument(
        "--base_bits",
        type=int,
        default=4,
        choices=[4, 16, 32],
        help="Base model precision: 4=NF4 quantization, 16=BF16, 32=float32 (default: 4)",
    )
    parser.add_argument(
        "--adapter_bits",
        type=int,
        default=16,
        choices=[4, 16, 32],
        help="LoRA adapter precision: 4=load LoRA A/B as NF4 Linear layers, 16=BF16, 32=float32 (default: 16)",
    )
    parser.add_argument(
        "--merge_adapter",
        action="store_true",
        help="Merge LoRA adapters into base model before evaluation (may improve compatibility)",
    )
    parser.add_argument(
        "--no_merge_loqt_adapter",
        action="store_true",
        help="Keep LoQT adapter_b dynamic during evaluation instead of merging it into static evaluation weights.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu",
        help="Tasks to evaluate (comma-separated for multiple, e.g., mmlu,hellaswag,gsm8k)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation per device",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply the tokenizer chat template during lm-eval-harness prompting.",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        help="Format few-shot examples as multi-turn chat messages when chat templating is enabled.",
    )

    args = parser.parse_args()

    run_evaluation(
        base_model=args.base_model,
        lora_model=args.lora_model,
        gradcodes_artifact=args.gradcodes_artifact,
        pvtuning_artifact=args.pvtuning_artifact,
        loqt_artifact=args.loqt_artifact,
        qalora_artifact=args.qalora_artifact,
        qzo_artifact=args.qzo_artifact,
        quzo_artifact=args.quzo_artifact,
        qes_artifact=args.qes_artifact,
        tasks=args.tasks,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        output_path=args.output,
        base_bits=args.base_bits,
        adapter_bits=args.adapter_bits,
        merge_adapter=args.merge_adapter,
        merge_loqt_adapter=not args.no_merge_loqt_adapter,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
    )


if __name__ == "__main__":
    main()
