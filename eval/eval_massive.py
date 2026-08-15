"""
Evaluate causal LM / LoRA / Gradcodes / LoQT / QA-LoRA checkpoints on MASSIVE.

This script mirrors the model-loading surface of ``eval_lm.py`` but runs
generation on ``AmazonScience/massive``, ``subset=en-US``, ``split=test`` using
the same prompt/gold formatting as ``train_massive_gradcodes.py``.

Reported metrics:
  * Exact Match: intent + slot multiset both match after normalization
  * Intent Accuracy
  * Slot F1: micro-F1 over (slot, value) pairs

Examples:
    CUDA_VISIBLE_DEVICES=0 python eval_massive.py \
        --qlora_checkpoint ./outputs/qwen-3-0.6b-qlora-massive/checkpoint-480 \
        --base_bits 4 \
        --adapter_bits 16 \
        --no-merge_adapter

    CUDA_VISIBLE_DEVICES=0 python eval_massive.py \
        --qlora_checkpoint ./outputs/qwen-3-0.6b-qalora-massive/epoch_0001 \
        --base_bits 4
    
    CUDA_VISIBLE_DEVICES=1,2,3,4 python eval_massive.py \
        --gradcodes_artifact ./outputs/qwen-3-0.6b-gradcodes-massive/epoch_0005 \
        --base_bits 4 

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch \
        --main_process_port 11011 \
        --num_processes 7 \
        eval_massive.py \
        --base_model meta-llama/Llama-3.2-3B-Instruct \
        --base_bits 4 \
        --generation_batch_size 16

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch \
        --main_process_port 11011 \
        --num_processes 7 \
         eval_massive.py \
        --qlora_checkpoint ./outputs/llama3.2-3b-instruct-qlora-massive/checkpoint-1236 \
        --base_bits 4 \
        --adapter_bits 16 \
        --generation_batch_size 16 \
        --merge_adapter

    CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
        --main_process_port 11011 \
        --num_processes 4 \
         eval_massive.py \
        --qalora_artifact ./outputs/qwen3-0.6b-qalora-massive/epoch_0004

Exact Match:    0.53227976
Intent Accuracy:0.80766644
Slot F1:        0.67687320

Exact Match:    0.59583053
Intent Accuracy:0.83994620
Slot F1:        0.71973054

Exact Match:    0.61163416
Intent Accuracy:0.84969738
Slot F1:        0.73212686

e1
Exact Match:    0.61869536
Intent Accuracy:0.85474109
Slot F1:        0.72809558

e2
Exact Match:    0.66677875
Intent Accuracy:0.87558843
Slot F1:        0.76795080

e3
Exact Match:    0.62642905
Intent Accuracy:0.83154001
Slot F1:        0.74031723

e4
Exact Match:    0.69166106
Intent Accuracy:0.88130464
Slot F1:        0.78741827

e5
Exact Match:    0.69367855
Intent Accuracy:0.88298588
Slot F1:        0.78620690

q306
e1
Exact Match:    0.48486886
Intent Accuracy:0.75756557
Slot F1:        0.60783582

e2
Exact Match:    0.59852051
Intent Accuracy:0.84095494
Slot F1:        0.70422025

e3
Exact Match:    0.63248151
Intent Accuracy:0.84936113
Slot F1:        0.74547429

e4
Exact Match:    0.66610625
Intent Accuracy:0.86852724
Slot F1:        0.75880661

e5
Exact Match:    0.65803631
Intent Accuracy:0.85810356
Slot F1:        0.75778175
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset

from eval_lm import load_model_with_peft, load_qalora_checkpoint, setup_distributed

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"#"meta-llama/Llama-3.2-1B-Instruct"

MASSIVE_PROMPT_TEMPLATE = """You are a mobile voice assistant semantic parser.
Extract the user's intent and slots from the utterance.

Output JSON only in the format:
{{
  "intent": "...",
  "slots": [
    {{"slot": "...", "value": "..."}}
  ]
}}

Rules:
- Use only information explicitly present in the utterance.
- Keep slot values exactly as they appear in the utterance.
- If there is no slot, return "slots": [].

User utterance: {utt}
"""


class SingleProcessState:
    def __init__(self, device: torch.device):
        self.device = device
        self.is_main_process = True
        self.num_processes = 1
        self.process_index = 0

    def wait_for_everyone(self) -> None:
        return None


def maybe_disable_unsupported_socks_proxy(*, is_main_process: bool) -> None:
    try:
        import socksio  # noqa: F401
        return
    except ImportError:
        pass

    removed: dict[str, str] = {}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.environ.get(key)
        if value and value.lower().startswith("socks"):
            removed[key] = value
            os.environ.pop(key, None)

    if removed and is_main_process:
        print(
            "Detected SOCKS proxy environment variables, but httpx[socks]/socksio is not installed; "
            "temporarily ignoring those proxy settings for this run."
        )
        for key, value in removed.items():
            print(f"  unset {key}={value}")


def is_accelerate_distributed_launch() -> bool:
    distributed_markers = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "ACCELERATE_PROCESS_INDEX")
    return any(name in os.environ for name in distributed_markers)


def single_process_device_from_arg(device_map: str) -> torch.device:
    normalized = str(device_map).strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        return torch.device(device_map)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_runtime_state(args: argparse.Namespace):
    if is_accelerate_distributed_launch():
        return setup_distributed()
    return SingleProcessState(device=single_process_device_from_arg(args.device_map))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HF / QLoRA / Gradcodes / LoQT / QA-LoRA models on MASSIVE en-US test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    model_group = parser.add_argument_group("model loading")
    model_group.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=(
            "Base model path/HF id. You may also pass a QLoRA adapter directory or "
            "a Gradcodes / LoQT / QA-LoRA checkpoint directory here and let the "
            "script auto-detect it."
        ),
    )
    model_group.add_argument(
        "--lora_model",
        type=str,
        default=None,
        help="Optional PEFT LoRA/QLoRA adapter directory or QA-LoRA checkpoint directory.",
    )
    model_group.add_argument(
        "--qlora_checkpoint",
        "--qlora-checkpoint",
        type=str,
        default=None,
        help=(
            "Explicit PEFT/QLoRA adapter checkpoint directory or QA-LoRA checkpoint directory. "
            "This is an alias for --lora_model, and the base model is inferred from checkpoint "
            "metadata when possible."
        ),
    )
    model_group.add_argument(
        "--gradcodes_artifact",
        type=str,
        default=None,
        help="Gradcodes artifact directory, Gradcodes checkpoint directory, or gradcodes_state.pt.",
    )
    model_group.add_argument(
        "--loqt_artifact",
        type=str,
        default=None,
        help="LoQT checkpoint directory or loqt_state.pt.",
    )
    model_group.add_argument(
        "--qalora_artifact",
        type=str,
        default=None,
        help="QA-LoRA checkpoint directory or qalora_state.pt.",
    )
    model_group.add_argument(
        "--base_bits",
        type=int,
        default=4,
        choices=[4, 16, 32],
        help="Base precision: 4=bitsandbytes NF4, 16=BF16, 32=float32.",
    )
    model_group.add_argument(
        "--adapter_bits",
        type=int,
        default=16,
        choices=[4, 16, 32],
        help="LoRA adapter precision: 4=NF4 LoRA A/B, 16=BF16, 32=float32.",
    )
    model_group.add_argument(
        "--merge_adapter",
        "--merge-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Merge dense LoRA/QLoRA adapters into the base model before generation. "
            "Use --no-merge_adapter to keep adapters dynamic."
        ),
    )
    model_group.add_argument("--no_merge_adapter", dest="merge_adapter", action="store_false", help=argparse.SUPPRESS)
    model_group.add_argument("--merge_adpater", dest="merge_adapter", action="store_true", help=argparse.SUPPRESS)
    model_group.add_argument("--no_merge_adpater", dest="merge_adapter", action="store_false", help=argparse.SUPPRESS)
    model_group.add_argument(
        "--no_merge_loqt_adapter",
        action="store_true",
        help="Keep LoQT adapter_b dynamic instead of merging it into static eval weights.",
    )
    model_group.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map passed to the loader outside accelerate multi-process mode.",
    )

    data_group = parser.add_argument_group("dataset")
    data_group.add_argument(
        "--dataset_name",
        type=str,
        default="AmazonScience/massive",
        help="Hugging Face dataset name.",
    )
    data_group.add_argument(
        "--dataset_config_name",
        type=str,
        default="en-US",
        help="Dataset config / subset name.",
    )
    data_group.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to evaluate.",
    )
    data_group.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Optional cap for a quick smoke test.",
    )

    generation_group = parser.add_argument_group("generation")
    generation_group.add_argument("--generation_batch_size", type=int, default=4, help="Batch size for model.generate.")
    generation_group.add_argument("--max_new_tokens", type=int, default=128, help="Maximum generated tokens per example.")
    generation_group.add_argument("--min_new_tokens", type=int, default=0, help="Minimum generated tokens per example.")
    generation_group.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sampling during generation. Defaults to greedy decoding.",
    )
    generation_group.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    generation_group.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling p.")
    generation_group.add_argument("--top_k", type=int, default=50, help="Top-k sampling. Use 0 to disable.")
    generation_group.add_argument("--repetition_penalty", type=float, default=1.0, help="Generation repetition penalty.")
    generation_group.add_argument(
        "--stop_sequences",
        type=str,
        default="",
        help="Comma-separated strings used to trim decoded outputs. Empty disables trimming.",
    )

    output_group = parser.add_argument_group("outputs")
    output_group.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional run name used to create the output directory.",
    )
    output_group.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to ./massive_eval_results/<name>.",
    )
    output_group.add_argument(
        "--overwrite_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Regenerate predictions even if predictions.json already exists. "
            "Defaults to true; pass --no-overwrite_predictions to reuse existing predictions."
        ),
    )

    return parser.parse_args()


def default_name_from_path(path: str) -> str:
    path_obj = Path(path)
    if path_obj.exists():
        return path_obj.name or path_obj.parent.name
    return path.replace("/", "_")


def resolve_peft_adapter_dir(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None

    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "adapter_config.json":
        candidate = candidate.parent
    if not candidate.is_dir():
        return None

    adapter_config_path = candidate / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    if any(
        (candidate / filename).exists()
        for filename in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "adapter_model.safetensors.index.json",
            "adapter_model.bin.index.json",
        )
    ):
        return candidate

    if any(candidate.glob("adapter_model-*.safetensors")) or any(candidate.glob("adapter_model-*.bin")):
        return candidate

    return None


def resolve_gradcodes_checkpoint(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None

    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "gradcodes_state.pt":
        return candidate
    if candidate.is_dir() and (candidate / "gradcodes_state.pt").exists():
        return candidate
    return None


def resolve_qalora_checkpoint(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None

    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "qalora_state.pt":
        return candidate
    if candidate.is_dir() and (candidate / "qalora_state.pt").exists():
        return candidate
    return None


def load_gradcodes_training_summary(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    summary_path = checkpoint_dir / "training_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def load_qalora_training_summary(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    summary_path = checkpoint_dir / "training_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def infer_qalora_base_model(checkpoint_path: Path) -> Optional[str]:
    summary = load_qalora_training_summary(checkpoint_path)
    base_model = summary.get("base_model_name_or_path")
    if isinstance(base_model, str) and base_model:
        return base_model

    payload = load_qalora_checkpoint(str(checkpoint_path))
    base_model = payload.get("base_model_name_or_path")
    if isinstance(base_model, str) and base_model:
        return base_model
    return None


def load_peft_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    with (adapter_dir / "adapter_config.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{adapter_dir / 'adapter_config.json'} is not a valid PEFT adapter config.")
    return payload


def normalize_qlora_checkpoint_args(args: argparse.Namespace, *, is_main_process: bool) -> argparse.Namespace:
    if args.qlora_checkpoint is not None:
        if args.lora_model is not None:
            raise ValueError("Pass a QLoRA adapter checkpoint through only one of --qlora_checkpoint or --lora_model.")
        args.lora_model = args.qlora_checkpoint

    base_as_adapter = resolve_peft_adapter_dir(args.base_model)
    base_as_qalora = resolve_qalora_checkpoint(args.base_model)
    lora_as_adapter = resolve_peft_adapter_dir(args.lora_model)
    lora_as_gradcodes = resolve_gradcodes_checkpoint(args.lora_model)
    lora_as_qalora = resolve_qalora_checkpoint(args.lora_model)
    explicit_qalora_artifact = resolve_qalora_checkpoint(args.qalora_artifact)

    if lora_as_qalora is not None:
        if any(path is not None for path in (args.gradcodes_artifact, args.loqt_artifact, args.qalora_artifact)):
            raise ValueError(
                "The checkpoint passed via --lora_model/--qlora_checkpoint is a QA-LoRA checkpoint, "
                "but another artifact flag is also set. Pass it through only one flag."
            )

        args.qalora_artifact = args.lora_model
        args.lora_model = None

        inferred_base_model = infer_qalora_base_model(lora_as_qalora)
        if inferred_base_model and args.base_model == DEFAULT_BASE_MODEL and inferred_base_model != args.base_model:
            args.base_model = inferred_base_model

        if is_main_process:
            print(
                "Detected a QA-LoRA checkpoint via --lora_model/--qlora_checkpoint; "
                f"using artifact {args.qalora_artifact}."
            )
            if inferred_base_model and args.base_model == inferred_base_model:
                print(f"Using base model from QA-LoRA checkpoint metadata: {args.base_model}")
            elif inferred_base_model:
                print(
                    "Warning: --base_model does not match the QA-LoRA checkpoint metadata "
                    f"({args.base_model} != {inferred_base_model})."
                )

        return args

    if explicit_qalora_artifact is not None:
        inferred_base_model = infer_qalora_base_model(explicit_qalora_artifact)
        if inferred_base_model and args.base_model == DEFAULT_BASE_MODEL and inferred_base_model != args.base_model:
            args.base_model = inferred_base_model
            if is_main_process:
                print(
                    "Inferred the QA-LoRA checkpoint base model from checkpoint metadata: "
                    f"{inferred_base_model}"
                )
        elif is_main_process and inferred_base_model and args.base_model != inferred_base_model:
            print(
                "Warning: --base_model does not match the QA-LoRA checkpoint metadata "
                f"({args.base_model} != {inferred_base_model})."
            )

    if lora_as_gradcodes is not None:
        if any(path is not None for path in (args.gradcodes_artifact, args.loqt_artifact, args.qalora_artifact)):
            raise ValueError(
                "The checkpoint passed via --lora_model/--qlora_checkpoint is a Gradcodes checkpoint, "
                "but another artifact flag is also set. Pass it through only one flag."
            )

        args.gradcodes_artifact = args.lora_model
        args.lora_model = None

        summary = load_gradcodes_training_summary(lora_as_gradcodes)
        inferred_base_model = summary.get("base_model_name_or_path")
        if inferred_base_model and args.base_model == DEFAULT_BASE_MODEL and inferred_base_model != args.base_model:
            args.base_model = inferred_base_model

        if is_main_process:
            print(
                "Detected a Gradcodes checkpoint via --lora_model/--qlora_checkpoint; "
                f"using artifact {args.gradcodes_artifact}."
            )
            if inferred_base_model:
                print(f"Using base model from training_summary.json: {args.base_model}")

        return args

    if base_as_qalora is not None:
        if args.lora_model is not None:
            raise ValueError(
                "--base_model points to a QA-LoRA checkpoint, but --lora_model/--qlora_checkpoint is also set. "
                "Pass the checkpoint through only one flag."
            )
        if any(path is not None for path in (args.gradcodes_artifact, args.loqt_artifact, args.qalora_artifact)):
            raise ValueError(
                "--base_model points to a QA-LoRA checkpoint while another artifact flag is also set. "
                "Pass the checkpoint through only one flag."
            )

        args.qalora_artifact = args.base_model
        inferred_base_model = infer_qalora_base_model(base_as_qalora)
        if not inferred_base_model:
            raise ValueError(
                f"Could not infer the base model from {base_as_qalora}. "
                "Pass --base_model explicitly and point --qalora_artifact at the checkpoint instead."
            )
        args.base_model = inferred_base_model

        if is_main_process:
            print(
                "Detected a QA-LoRA checkpoint via --base_model; "
                f"using artifact {args.qalora_artifact} on base model {inferred_base_model}."
            )
        return args

    if base_as_adapter is not None and any(
        path is not None for path in (args.gradcodes_artifact, args.loqt_artifact, args.qalora_artifact)
    ):
        raise ValueError(
            "--base_model points to a PEFT/QLoRA checkpoint while a quantized artifact flag is also set. "
            "Pass the QLoRA checkpoint via --lora_model or remove the artifact flag."
        )

    if base_as_adapter is not None and args.lora_model is not None:
        raise ValueError(
            "--base_model points to a PEFT/QLoRA checkpoint, but --lora_model is also set. "
            "Pass the adapter checkpoint through only one flag."
        )

    if base_as_adapter is not None:
        adapter_config = load_peft_adapter_config(base_as_adapter)
        inferred_base_model = adapter_config.get("base_model_name_or_path")
        if not inferred_base_model:
            raise ValueError(
                f"Could not infer the base model from {base_as_adapter / 'adapter_config.json'}. "
                "Pass --base_model explicitly and point --lora_model at the checkpoint instead."
            )
        args.base_model = inferred_base_model
        args.lora_model = str(base_as_adapter)
        if is_main_process:
            print(
                "Detected a PEFT/QLoRA checkpoint via --base_model; "
                f"using adapter {base_as_adapter} on base model {inferred_base_model}."
            )
        return args

    if lora_as_adapter is None:
        if args.lora_model is not None and Path(args.lora_model).expanduser().exists():
            raise ValueError(
                f"{args.lora_model} is a local path, but it is not a PEFT/QLoRA adapter checkpoint "
                "with adapter_config.json + adapter_model.*, and it is not a QA-LoRA/Gradcodes "
                "checkpoint directory. For QA-LoRA checkpoints, pass --qalora_artifact or point "
                "--base_model at the epoch directory."
            )
        return args

    adapter_config = load_peft_adapter_config(lora_as_adapter)
    inferred_base_model = adapter_config.get("base_model_name_or_path")
    if not inferred_base_model:
        if args.base_model:
            return args
        raise ValueError(
            f"Could not infer the base model from {lora_as_adapter / 'adapter_config.json'}. "
            "Pass --base_model explicitly."
        )

    if args.base_model == DEFAULT_BASE_MODEL and inferred_base_model != args.base_model:
        args.base_model = inferred_base_model
        if is_main_process:
            print(
                "Inferred the QLoRA checkpoint base model from adapter_config.json: "
                f"{inferred_base_model}"
            )
    elif is_main_process and args.base_model != inferred_base_model:
        print(
            "Warning: --base_model does not match the QLoRA checkpoint's adapter_config.json "
            f"({args.base_model} != {inferred_base_model})."
        )

    return args


def get_output_dir(args: argparse.Namespace, name: str) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return SCRIPT_DIR / "massive_eval_results" / name


def default_eval_name(args: argparse.Namespace) -> str:
    for path in (
        args.lora_model,
        args.gradcodes_artifact,
        args.loqt_artifact,
        args.qalora_artifact,
        args.base_model,
    ):
        if path:
            return default_name_from_path(path)
    return "massive_eval"


def _resolve_hf_api_bases() -> list[str]:
    api_bases: list[str] = []
    endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    if endpoint:
        api_bases.append(endpoint)
    if "https://huggingface.co" not in api_bases:
        api_bases.append("https://huggingface.co")
    return api_bases


def _load_dataset_via_parquet_api(
    dataset_name: str,
    dataset_config_name: str,
    dataset_split: str,
):
    last_error: Optional[Exception] = None

    for api_base in _resolve_hf_api_bases():
        parquet_index_url = (
            f"{api_base}/api/datasets/{dataset_name}/parquet/"
            f"{dataset_config_name}/{dataset_split}"
        )
        try:
            with urllib.request.urlopen(parquet_index_url, timeout=30) as response:
                parquet_urls = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            continue

        if not isinstance(parquet_urls, list) or not parquet_urls:
            last_error = RuntimeError(
                f"Parquet API returned no files for {dataset_name}/{dataset_config_name} [{dataset_split}] "
                f"from {parquet_index_url}"
            )
            continue

        return load_dataset(
            "parquet",
            data_files={dataset_split: parquet_urls},
            split=dataset_split,
        )

    raise RuntimeError(
        f"Failed to load parquet fallback for {dataset_name}/{dataset_config_name} [{dataset_split}]"
    ) from last_error


def _looks_like_slot_name(text: str) -> bool:
    stripped = text.strip()
    return bool(re.fullmatch(r"[A-Za-z0-9_]+", stripped)) and ("_" in stripped or stripped.islower())


def _parse_massive_slots(utterance: str, annotated_utterance: Optional[object]) -> list[dict[str, str]]:
    if annotated_utterance is None or not isinstance(annotated_utterance, str):
        return []

    slots: list[dict[str, str]] = []
    for annotation in re.findall(r"\[([^\[\]]+)\]", annotated_utterance):
        if ":" not in annotation:
            continue

        left, right = re.split(r"\s*:\s*", annotation, maxsplit=1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            continue

        left_in_utterance = left in utterance
        right_in_utterance = right in utterance
        if left_in_utterance and not right_in_utterance:
            value = left
            slot_name = right
        elif right_in_utterance and not left_in_utterance:
            value = right
            slot_name = left
        elif _looks_like_slot_name(left) and not _looks_like_slot_name(right):
            slot_name = left
            value = right
        elif _looks_like_slot_name(right) and not _looks_like_slot_name(left):
            slot_name = right
            value = left
        else:
            slot_name = left
            value = right

        slots.append({"slot": slot_name, "value": value})

    return slots


def _intent_to_text(intent_value: object, intent_label_names: Optional[list[str]]) -> str:
    if isinstance(intent_value, str):
        return intent_value
    if isinstance(intent_value, int) and intent_label_names is not None and 0 <= intent_value < len(intent_label_names):
        return intent_label_names[intent_value]
    return "" if intent_value is None else str(intent_value)


def normalize_slot(slot: object) -> Optional[dict[str, str]]:
    if isinstance(slot, dict):
        slot_name = slot.get("slot")
        value = slot.get("value")
    elif isinstance(slot, (list, tuple)) and len(slot) == 2:
        slot_name, value = slot
    else:
        return None

    slot_name = "" if slot_name is None else str(slot_name).strip()
    value = "" if value is None else str(value).strip()
    if not slot_name or not value:
        return None
    return {"slot": slot_name, "value": value}


def normalize_semantic_frame(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"intent": "", "slots": []}

    intent = payload.get("intent", "")
    intent_text = "" if intent is None else str(intent).strip()

    raw_slots = payload.get("slots", [])
    normalized_slots: list[dict[str, str]] = []
    if isinstance(raw_slots, list):
        for slot in raw_slots:
            normalized = normalize_slot(slot)
            if normalized is not None:
                normalized_slots.append(normalized)

    normalized_slots.sort(key=lambda item: (item["slot"], item["value"]))
    return {"intent": intent_text, "slots": normalized_slots}


def build_gold_frame(row: dict[str, Any], intent_label_names: Optional[list[str]]) -> dict[str, Any]:
    utterance = "" if row.get("utt") is None else str(row["utt"])
    return normalize_semantic_frame(
        {
            "intent": _intent_to_text(row.get("intent"), intent_label_names),
            "slots": _parse_massive_slots(utterance, row.get("annot_utt")),
        }
    )


def load_massive_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=args.dataset_split,
            trust_remote_code=True,
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        dataset = _load_dataset_via_parquet_api(
            args.dataset_name,
            args.dataset_config_name,
            args.dataset_split,
        )

    if args.max_instances is not None:
        dataset = dataset.select(range(min(len(dataset), args.max_instances)))

    intent_label_names = None
    intent_feature = getattr(dataset, "features", {}).get("intent")
    if hasattr(intent_feature, "names"):
        intent_label_names = list(intent_feature.names)

    records: list[dict[str, Any]] = []
    for row in dataset:
        row_dict = dict(row)
        utterance = "" if row_dict.get("utt") is None else str(row_dict["utt"])
        records.append(
            {
                "id": row_dict.get("id"),
                "utt": utterance,
                "annot_utt": row_dict.get("annot_utt"),
                "prompt": MASSIVE_PROMPT_TEMPLATE.format(utt=utterance),
                "gold": build_gold_frame(row_dict, intent_label_names),
            }
        )

    return records


def get_generation_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            if device not in ("cpu", "disk", "meta"):
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def parse_stop_sequences(value: str) -> list[str]:
    if not value:
        return []
    return [item for item in (part.strip() for part in value.split(",")) if item]


def trim_stop_sequences(text: str, stop_sequences: list[str]) -> str:
    cut = None
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx >= 0:
            cut = idx if cut is None else min(cut, idx)
    if cut is not None:
        text = text[:cut]
    return text.strip()


def generation_kwargs_from_args(args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
        if args.top_k > 0:
            kwargs["top_k"] = args.top_k
    return kwargs


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return stripped


def extract_first_json_object(text: str) -> Optional[str]:
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for end_idx in range(start_idx, len(text)):
            curr = text[end_idx]
            if in_string:
                if escape:
                    escape = False
                elif curr == "\\":
                    escape = True
                elif curr == '"':
                    in_string = False
                continue

            if curr == '"':
                in_string = True
            elif curr == "{":
                depth += 1
            elif curr == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx : end_idx + 1]
    return None


def parse_generated_frame(text: str) -> tuple[dict[str, Any], Optional[str]]:
    cleaned = strip_code_fences(text)
    candidates: list[str] = []

    extracted = extract_first_json_object(cleaned)
    if extracted is not None:
        candidates.append(extracted)
    candidates.append(cleaned)

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        try:
            return normalize_semantic_frame(json.loads(candidate)), None
        except json.JSONDecodeError:
            pass

        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            return normalize_semantic_frame(parsed), "parsed_with_literal_eval"

    return {"intent": "", "slots": []}, "invalid_json"


def slot_counter(frame: dict[str, Any]) -> Counter[tuple[str, str]]:
    return Counter((slot["slot"], slot["value"]) for slot in frame.get("slots", []))


def compute_metrics(
    predictions: list[dict[str, Any]],
    *,
    dataset_name: str,
    dataset_config_name: str,
    dataset_split: str,
) -> dict[str, Any]:
    total = len(predictions)
    exact_match = 0
    intent_correct = 0
    slot_tp = 0
    slot_fp = 0
    slot_fn = 0
    parse_errors = 0

    for row in predictions:
        gold = normalize_semantic_frame(row.get("gold", {}))
        pred = normalize_semantic_frame(row.get("prediction", {}))
        if row.get("parse_error"):
            parse_errors += 1

        gold_slots = slot_counter(gold)
        pred_slots = slot_counter(pred)

        if pred.get("intent", "") == gold.get("intent", ""):
            intent_correct += 1
        if pred.get("intent", "") == gold.get("intent", "") and pred_slots == gold_slots:
            exact_match += 1

        all_slot_keys = set(gold_slots) | set(pred_slots)
        for key in all_slot_keys:
            tp = min(gold_slots.get(key, 0), pred_slots.get(key, 0))
            slot_tp += tp
            slot_fp += max(pred_slots.get(key, 0) - tp, 0)
            slot_fn += max(gold_slots.get(key, 0) - tp, 0)

    slot_precision = slot_tp / (slot_tp + slot_fp) if (slot_tp + slot_fp) > 0 else 0.0
    slot_recall = slot_tp / (slot_tp + slot_fn) if (slot_tp + slot_fn) > 0 else 0.0
    if slot_precision + slot_recall == 0:
        slot_f1 = 0.0
    else:
        slot_f1 = 2 * slot_precision * slot_recall / (slot_precision + slot_recall)

    return {
        "dataset_name": dataset_name,
        "dataset_config_name": dataset_config_name,
        "dataset_split": dataset_split,
        "num_examples": total,
        "exact_match": exact_match / total if total else 0.0,
        "intent_accuracy": intent_correct / total if total else 0.0,
        "slot_f1": slot_f1,
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "parse_error_rate": parse_errors / total if total else 0.0,
        "parse_errors": parse_errors,
        "slot_tp": slot_tp,
        "slot_fp": slot_fp,
        "slot_fn": slot_fn,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("MASSIVE Evaluation Results")
    print("=" * 80)
    print(f"Dataset: {metrics['dataset_name']} / {metrics['dataset_config_name']} / {metrics['dataset_split']}")
    print(f"Examples: {metrics['num_examples']}")
    print(f"Exact Match:    {metrics['exact_match']:.8f}")
    print(f"Intent Accuracy:{metrics['intent_accuracy']:.8f}")
    print(f"Slot F1:        {metrics['slot_f1']:.8f}")
    print(f"Slot Precision: {metrics['slot_precision']:.8f}")
    print(f"Slot Recall:    {metrics['slot_recall']:.8f}")
    print(f"Parse Error:    {metrics['parse_error_rate']:.8f} ({metrics['parse_errors']})")
    print("=" * 80)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_predictions(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_dir: Path,
    distributed_state=None,
) -> Optional[Path]:
    runtime_state = distributed_state
    if (
        not is_accelerate_distributed_launch()
        and torch.cuda.is_available()
        and str(args.device_map).strip().lower() != "cpu"
        and getattr(runtime_state, "device", torch.device("cpu")).type == "cpu"
    ):
        runtime_state = SingleProcessState(device=single_process_device_from_arg(args.device_map))

    process_index = getattr(runtime_state, "process_index", 0) if runtime_state is not None else 0
    num_processes = getattr(runtime_state, "num_processes", 1) if runtime_state is not None else 1
    is_main_process = runtime_state is None or runtime_state.is_main_process

    model, tokenizer = load_model_with_peft(
        base_model_path=args.base_model,
        lora_model_path=args.lora_model,
        gradcodes_artifact_path=args.gradcodes_artifact,
        loqt_artifact_path=args.loqt_artifact,
        qalora_artifact_path=args.qalora_artifact,
        base_bits=args.base_bits,
        adapter_bits=args.adapter_bits,
        device_map=args.device_map,
        merge_adapter=args.merge_adapter,
        merge_loqt_adapter=not args.no_merge_loqt_adapter,
        distributed_state=runtime_state,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if is_main_process:
        print("=" * 80)
        print("Generating MASSIVE predictions")
        print("=" * 80)
        print(f"Examples: {len(records)}")
        print(f"Generation batch size: {args.generation_batch_size}")
        print(f"Max new tokens: {args.max_new_tokens}")
        print(f"Distributed generation: {num_processes} process(es)")
        print("=" * 80)

    indexed_records = list(enumerate(records))
    local_records = indexed_records[process_index::num_processes]
    stop_sequences = parse_stop_sequences(args.stop_sequences)
    device = get_generation_device(model)
    gen_kwargs = generation_kwargs_from_args(args, tokenizer)
    local_outputs: list[dict[str, Any]] = []

    for batch_no, batch in enumerate(batched(local_records, args.generation_batch_size), start=1):
        batch_indices = [idx for idx, _ in batch]
        batch_rows = [row for _, row in batch]
        prompts = [row["prompt"] for row in batch_rows]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        completions = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)

        for row_index, row, completion in zip(batch_indices, batch_rows, completions):
            trimmed = trim_stop_sequences(completion, stop_sequences)
            parsed_prediction, parse_error = parse_generated_frame(trimmed)
            local_outputs.append(
                {
                    "id": row["id"],
                    "utt": row["utt"],
                    "annot_utt": row["annot_utt"],
                    "prompt": row["prompt"],
                    "gold": row["gold"],
                    "raw_output": completion,
                    "trimmed_output": trimmed,
                    "prediction": parsed_prediction,
                    "parse_error": parse_error,
                    "_index": row_index,
                }
            )

        if is_main_process and (batch_no == 1 or batch_no % 10 == 0):
            done = min(batch_no * args.generation_batch_size * num_processes, len(records))
            print(f"Generated about {done}/{len(records)} examples...")

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"predictions.rank{process_index:05d}.json"
    save_json(shard_path, local_outputs)

    if runtime_state is not None:
        runtime_state.wait_for_everyone()

    if not is_main_process:
        return None

    all_outputs: list[dict[str, Any]] = []
    for rank in range(num_processes):
        curr_shard = shard_dir / f"predictions.rank{rank:05d}.json"
        if not curr_shard.exists():
            raise FileNotFoundError(f"Missing prediction shard: {curr_shard}")
        all_outputs.extend(load_json(curr_shard))

    all_outputs.sort(key=lambda row: row["_index"])
    for row in all_outputs:
        row.pop("_index", None)

    predictions_path = output_dir / "predictions.json"
    save_json(predictions_path, all_outputs)
    print(f"Saved MASSIVE predictions to {predictions_path}")
    return predictions_path


def main() -> None:
    args = parse_args()
    distributed_state = build_runtime_state(args)
    is_main_process = distributed_state is None or distributed_state.is_main_process
    maybe_disable_unsupported_socks_proxy(is_main_process=is_main_process)
    args = normalize_qlora_checkpoint_args(args, is_main_process=is_main_process)
    if is_main_process and args.merge_adapter and args.lora_model is None:
        print(
            "merge_adapter was requested, but no LoRA/QLoRA adapter checkpoint is loaded; "
            "the flag has no effect for artifact-only checkpoints such as Gradcodes."
        )

    if args.name is not None:
        name = args.name
    else:
        name = default_eval_name(args)

    output_dir = get_output_dir(args, name)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    metrics_path = output_dir / "metrics.json"

    if predictions_path.exists() and not args.overwrite_predictions:
        if is_main_process:
            print(f"Reusing existing predictions at {predictions_path} because --no-overwrite_predictions was set.")
    else:
        records = load_massive_records(args)
        predictions_path = generate_predictions(
            args=args,
            records=records,
            output_dir=output_dir,
            distributed_state=distributed_state,
        )

    if distributed_state is not None:
        distributed_state.wait_for_everyone()

    if not is_main_process:
        return

    if predictions_path is None:
        predictions_path = output_dir / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Could not find predictions at {predictions_path}")

    predictions = load_json(predictions_path)
    if not isinstance(predictions, list):
        raise ValueError(f"{predictions_path} must contain a list of prediction records.")

    metrics = compute_metrics(
        predictions,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
    )
    save_json(metrics_path, metrics)
    print_metrics(metrics)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
