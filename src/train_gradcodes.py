"""
Train a causal language model on GSM8K, Alpaca, or MASSIVE with Gradcodes.

Compared with `train_gsm8k.py`, this script does not use LoRA + AdamW.
Instead, it performs:

1. stage-wise low-rank block expansion,
2. STE-guided proposal generation,
3. inverse-distance discrete sampling on integer factors,
4. merge-only acceptance on the discrete objective.

The implementation is intended as a practical research framework rather than a
drop-in reproduction of every training detail from the paper. By default it
quantizes all linear layers into the Gradcodes wrapper, while only the
configured target layers participate in discrete search.

CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 src/train_gradcodes.py --dataset gsm8k
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
import warnings
from typing import Dict, Iterable, List, Optional, Union, Tuple
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from gradcodes import (
    GradcodesLinear,
    build_inverse_distance_lattice,
    candidate_probability_under_distribution,
    collect_gradcodes_state,
    inverse_distance_probabilities_from_log_distances,
    iter_gradcodes_modules,
    load_gradcodes_state,
    replace_linear_with_gradcodes,
    sample_from_inverse_distance_distribution,
    scaled_proposal_step,
    select_nearest_from_inverse_distance_distribution,
    summarize_coordinate_probabilities,
)
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
DEFAULT_QUANTIZED_MODULES = DEFAULT_TARGET_MODULES
EPOCH_WALL_CLOCK_TIMES_FILENAME = "epoch_wall_clock_times.json"
SUPPORTED_QUANT_TYPES = ("nf4", "uniform", "int4", "mxfp4")
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

DATASET_DEFAULTS = {
    "gsm8k": {
        "dataset_name": "openai/gsm8k",
        "dataset_config_name": "main",
        "dataset_split": "train",
        "output_dir": "./outputs/llama-3.2-1b-gradcodes-gsm8k-nf4",
    },
    "alpaca": {
        "dataset_name": "yahma/alpaca-cleaned",
        "dataset_config_name": None,
        "dataset_split": "train",
        "output_dir": "./outputs/llama-3.2-1b-gradcodes-alpaca",
    },
    "massive": {
        "dataset_name": "AmazonScience/massive",
        "dataset_config_name": "en-US",
        "dataset_split": "train",
        "output_dir": "./outputs/llama-3.2-1b-gradcodes-massive",
    },
}


def parse_norm_p(value: str) -> float:
    value = value.strip().lower()
    if value in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    return float(value)


def parse_args(default_dataset: str = "gsm8k") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradcodes fine-tuning for GSM8K, Alpaca, or MASSIVE")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",#"meta-llama/Llama-3.2-1B-Instruct",
        help="Base model path or HF identifier.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory used for Gradcodes checkpoints and logs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        choices=tuple(DATASET_DEFAULTS),
        help="Dataset profile. It selects the required prompt/response preprocessing function.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to an epoch/final checkpoint directory or gradcodes_state.pt. "
            "--num_train_epochs is interpreted as the total target epoch budget."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Optional Hugging Face dataset override for the selected --dataset profile.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Optional Hugging Face dataset config override for the selected --dataset profile.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="Optional dataset split override for the selected --dataset profile.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=768,
        help="Maximum sequence length after tokenization.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size used for each discrete-search evaluation.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=4.0,
        help="Used to derive the total search budget when --stage_steps is unset.",
    )
    parser.add_argument(
        "--stage_steps",
        type=int,
        default=None,
        help="Number of inner-loop search steps per rank stage.",
    )
    parser.add_argument(
        "--num_stages",
        type=int,
        default=1,
        help="Number of stage-wise search blocks to run. Defaults to a legacy total-rank budget of 4 when unset.",
    )
    parser.add_argument(
        "--ranks_per_stage",
        type=int,
        default=8,
        help="Number of rank-one factors optimized jointly inside each stage.",
    )
    parser.add_argument(
        "--target_rank",
        type=int,
        default=None,
        help="Legacy total low-rank budget used only when --num_stages is unset.",
    )
    parser.add_argument(
        "--candidate_batch_size",
        type=int,
        default=16,
        help="Number of candidate models evaluated per step, including the current one.",
    )
    parser.add_argument(
        "--proposal_lr_a",
        type=float,
        default=3,
        help="Proposal learning rate for the output-side rank factor.",
    )
    parser.add_argument(
        "--proposal_lr_b",
        type=float,
        default=3,
        help="Proposal learning rate for the input-side rank factor.",
    )
    parser.add_argument(
        "--proposal_lr_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine"],
        help="Scheduler applied to proposal_lr_a/proposal_lr_b across all inner search steps.",
    )
    parser.add_argument(
        "--proposal_lr_warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup ratio for the proposal learning-rate schedule.",
    )
    parser.add_argument(
        "--proposal_lr_min_ratio",
        type=float,
        default=0.0,
        help="Minimum multiplier reached by the proposal learning-rate schedule after decay.",
    )
    parser.add_argument(
        "--lattice_weight_decay",
        "--weight_decay",
        dest="lattice_weight_decay",
        type=float,
        default=0.5,
        help=(
            "L2 regularization coefficient added directly to the lattice search objective. "
            "Scale optimization should use its own optimizer-level weight decay instead."
        ),
    )
    parser.add_argument(
        "--scale_learning_rate",
        type=float,
        default=5e-5,#0.0002,
        help="AdamW learning rate used for block-shared scale updates. Set to 0 to disable scale optimization.",
    )
    parser.add_argument(
        "--scale_weight_decay",
        type=float,
        default=0.00,
        help="AdamW weight decay applied to the learnable block-shared scale parameters.",
    )
    parser.add_argument(
        "--scale_adam_beta1",
        type=float,
        default=0.9,
        help="AdamW beta1 for scale updates.",
    )
    parser.add_argument(
        "--scale_adam_beta2",
        type=float,
        default=0.999,
        help="AdamW beta2 for scale updates.",
    )
    parser.add_argument(
        "--scale_adam_epsilon",
        type=float,
        default=1e-8,
        help="AdamW epsilon for scale updates.",
    )
    parser.add_argument(
        "--scale_max_grad_norm",
        type=float,
        default=0.0,
        help="Optional gradient clipping threshold for scale updates. Use 0 to disable.",
    )
    parser.add_argument(
        "--proposal_epsilon",
        type=float,
        default=1e-8,
        help="Smoothing constant for inverse-distance sampling.",
    )
    parser.add_argument(
        "--proposal_tau",
        type=float,
        default=2.0,
        help="Steepness of the inverse-distance proposal distribution: p is proportional to (distance + epsilon)^(-tau). Use -1 to enable deterministic top-k one-step grid search.",
    )
    parser.add_argument(
        "--proposal_p",
        type=float,
        default=1/2,
        help=(
            "When > 0, adapt tau at each search step so the joint probability of sampling "
            "the current state equals p across all coordinates. Overrides --proposal_tau."
        ),
    )
    parser.add_argument(
        "--top_k",
        "--proposal_top_k",
        dest="top_k",
        type=int,
        default=1,
        help="When --proposal_tau=-1, choose the top-k |lr * guide| coordinates, move each by one lattice step along its sign direction, and enumerate all 2^k subset candidates.",
    )
    parser.add_argument(
        "--min_step_norm",
        type=float,
        default=0.0,
        help="Optional p-norm floor for each A/B proposal step. Set to 0 to disable norm boosting.",
    )
    parser.add_argument(
        "--max_step_norm",
        type=float,
        default=0.5,
        help="Optional p-norm cap for each A/B proposal step. Set to 0 to disable norm clipping.",
    )
    parser.add_argument(
        "--norm_p",
        type=parse_norm_p,
        default=float("inf"),
        help="p used by --min_step_norm/--max_step_norm when scaling each A/B proposal step. Use 'inf' for max norm.",
    )
    parser.add_argument(
        "--quant_bits",
        type=int,
        default=4,
        help="Bit-width of the fixed low-bit lattice. NF4 currently requires 4 bits.",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="nf4",
        choices=SUPPORTED_QUANT_TYPES,
        help=(
            "Discrete quantizer used inside Gradcodes. int4 is signed 4-bit "
            "uniform quantization; mxfp4 uses FP4 E2M1 values with power-of-two block scales."
        ),
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help=(
            "Scale-block size along the input axis for each wrapped linear layer. "
            "Defaults to 32 for mxfp4 and 64 otherwise."
        ),
    )
    parser.add_argument(
        "--amax",
        type=int,
        default=1,
        help="Half-width of the local integer proposal window for the output-side rank factor.",
    )
    parser.add_argument(
        "--bmax",
        type=int,
        default=1,
        help="Half-width of the local integer proposal window for the input-side rank factor.",
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=DEFAULT_TARGET_MODULES,
        help="Linear module suffixes that participate in Gradcodes search.",
    )
    parser.add_argument(
        "--quantized_modules",
        nargs="+",
        default=DEFAULT_QUANTIZED_MODULES,
        help="Linear module suffixes to quantize with Gradcodes wrappers. Defaults to all linear layers.",
    )
    parser.add_argument(
        "--power_iterations",
        type=int,
        default=8,
        help="Power-iteration steps for gradient-based stage initialization.",
    )
    parser.add_argument(
        "--stage_init_method",
        type=str,
        default="lora",
        choices=["lora", "gradient"],
        help="Initialization method for the active stage: LoRA-style random A with zero B, or gradient/SVD-based initialization.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--gradient_capture_dtype",
        type=str,
        default="float32",
        choices=["model", "float32"],
        help=(
            "Dtype used for the temporary full-weight variables and their gradients during "
            "Gradcodes guidance. 'model' is substantially more memory efficient; float32 "
            "preserves the legacy behavior."
        ),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Logging interval in search steps.",
    )
    parser.add_argument(
        "--save_every_stage",
        action="store_true",
        help="Deprecated. Stage-level search-state checkpoints were replaced by epoch HF saves.",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Pass flash attention v2 to `from_pretrained` when supported.",
    )
    checkpointing_group = parser.add_mutually_exclusive_group()
    checkpointing_group.add_argument(
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on backward passes (default).",
    )
    checkpointing_group.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing.",
    )
    parser.set_defaults(gradient_checkpointing=True)

    return parser.parse_args()


def resolve_dataset_defaults(args: argparse.Namespace) -> None:
    """Fill profile defaults after CLI parsing, preserving explicit user overrides."""
    defaults = DATASET_DEFAULTS[args.dataset]
    for field in ("dataset_name", "dataset_config_name", "dataset_split", "output_dir"):
        if getattr(args, field) is None:
            setattr(args, field, defaults[field])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda_available():
        torch.cuda.manual_seed_all(seed)


def distributed_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_world_size() -> int:
    return dist.get_world_size() if distributed_enabled() else 1


def distributed_rank() -> int:
    return dist.get_rank() if distributed_enabled() else 0


def is_main_process() -> bool:
    return distributed_rank() == 0


def initialize_distributed() -> Tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return 1, local_rank

    if not dist.is_initialized():
        backend = "nccl" if cuda_available() else "gloo"
        dist.init_process_group(backend=backend)
    return dist.get_world_size(), local_rank


def finalize_distributed() -> None:
    if distributed_enabled():
        dist.barrier()
        dist.destroy_process_group()


def distributed_mean_scalar(value: Union[float, torch.Tensor], device: torch.device) -> float:
    tensor = value.detach().to(device=device, dtype=torch.float32) if isinstance(value, torch.Tensor) else torch.tensor(
        float(value),
        device=device,
        dtype=torch.float32,
    )
    if distributed_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(distributed_world_size())
    return float(tensor.item())


def distributed_max_scalar(value: Union[float, torch.Tensor], device: torch.device) -> float:
    tensor = value.detach().to(device=device, dtype=torch.float32) if isinstance(value, torch.Tensor) else torch.tensor(
        float(value),
        device=device,
        dtype=torch.float32,
    )
    if distributed_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def distributed_mean_tensor_values(values: List[torch.Tensor], device: torch.device) -> List[float]:
    if not values:
        return []

    tensor = torch.stack([value.detach().to(device=device, dtype=torch.float32) for value in values], dim=0)
    if distributed_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(distributed_world_size())
    return [float(entry.item()) for entry in tensor]


def sync_gradcodes_gradients(modules: List[GradcodesLinear]) -> None:
    if not distributed_enabled():
        return

    world_size = distributed_world_size()
    for module in modules:
        if module.last_weight is None or module.last_weight.grad is None:
            continue
        dist.all_reduce(module.last_weight.grad, op=dist.ReduceOp.SUM)
        module.last_weight.grad.div_(world_size)


def synchronize_wall_clock_timer(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_search_gradient_capture(modules: List[GradcodesLinear], enabled: bool) -> None:
    for module in modules:
        module.set_capture_weight_gradients(enabled)


def iter_scale_parameters(modules: List[GradcodesLinear]) -> Iterable[torch.nn.Parameter]:
    for module in modules:
        if module.scale_log_factors.requires_grad:
            yield module.scale_log_factors


def count_scale_parameters(modules: List[GradcodesLinear]) -> int:
    return sum(parameter.numel() for parameter in iter_scale_parameters(modules))


def scale_gradient_norm(parameters: List[torch.nn.Parameter]) -> float:
    grad_tensors = [
        parameter.grad.detach().to(torch.float32).reshape(-1)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not grad_tensors:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(grad_tensors), ord=2).item())


def sync_scale_gradients(modules: List[GradcodesLinear]) -> None:
    if not distributed_enabled():
        return

    world_size = distributed_world_size()
    for parameter in iter_scale_parameters(modules):
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def build_scale_optimizer(
    modules: List[GradcodesLinear],
    args: argparse.Namespace,
) -> Optional[torch.optim.Optimizer]:
    if args.scale_learning_rate <= 0.0:
        return None

    scale_parameters = list(iter_scale_parameters(modules))
    if not scale_parameters:
        return None

    return torch.optim.AdamW(
        scale_parameters,
        lr=args.scale_learning_rate,
        betas=(args.scale_adam_beta1, args.scale_adam_beta2),
        eps=args.scale_adam_epsilon,
        weight_decay=args.scale_weight_decay,
    )


def should_skip_search_update(
    modules: List[GradcodesLinear],
    *,
    proposal_lr_a: float,
    proposal_lr_b: float,
) -> bool:
    if not modules:
        return True
    if all(module.full_matrix_mode for module in modules):
        return proposal_lr_a <= 0.0
    return proposal_lr_a <= 0.0 and proposal_lr_b <= 0.0


def serialize_candidate_state_for_broadcast(
    candidates: List[List[Tuple[torch.Tensor, torch.Tensor]]],
) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    serialized: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    for candidate in candidates:
        serialized_candidate: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for a, b in candidate:
            serialized_candidate.append((a.detach().cpu(), b.detach().cpu()))
        serialized.append(serialized_candidate)
    return serialized


def serialize_single_candidate_state_for_broadcast(
    candidate: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    serialized_candidate: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for a, b in candidate:
        serialized_candidate.append((a.detach().cpu(), b.detach().cpu()))
    return serialized_candidate


def synchronize_active_stage_from_main(modules: List[GradcodesLinear]) -> None:
    if not distributed_enabled():
        return

    payload = [serialize_single_candidate_state_for_broadcast(snapshot_model_candidate(modules))] if is_main_process() else [None]
    broadcast_device = modules[0].storage_device if modules else None
    dist.broadcast_object_list(payload, src=0, device=broadcast_device)
    apply_model_candidate(modules, payload[0])


def cuda_available() -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False


def torch_dtype_from_arg(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def tokenize_prompt_response_pairs(
    prompt_texts: List[str],
    response_texts: List[str],
    tokenizer,
    max_length: int,
) -> Dict[str, List[List[int]]]:
    """Tokenize examples and mask the prompt so loss is computed on responses only."""
    eos = tokenizer.eos_token or ""
    full_texts = [prompt + response + eos for prompt, response in zip(prompt_texts, response_texts)]

    tokenized_full = tokenizer(full_texts, truncation=True, max_length=max_length, padding=False)
    tokenized_prompt = tokenizer(prompt_texts, truncation=True, max_length=max_length, padding=False)

    labels: List[List[int]] = []
    for input_ids, attention_mask, prompt_attention_mask in zip(
        tokenized_full["input_ids"],
        tokenized_full["attention_mask"],
        tokenized_prompt["attention_mask"],
    ):
        label = list(input_ids)
        prompt_len = int(sum(prompt_attention_mask))
        for idx in range(min(prompt_len, len(label))):
            label[idx] = -100
        for idx, mask in enumerate(attention_mask):
            if mask == 0:
                label[idx] = -100
        labels.append(label)

    tokenized_full["labels"] = labels
    return tokenized_full


def preprocess_gsm8k(examples: Dict[str, List[str]], tokenizer, max_length: int) -> Dict[str, List[List[int]]]:
    """Response-only GSM8K formatting, matching the original Gradcodes script."""
    questions = examples.get("question", [""] * len(examples["answer"]))
    answers = examples["answer"]
    return tokenize_prompt_response_pairs(
        [f"Q: {question}\nA: Let's think step by step. " for question in questions],
        [answer if answer is not None else "" for answer in answers],
        tokenizer,
        max_length,
    )


def format_alpaca_prompt(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    prompt_instruction = f"{instruction}\n\n{input_text}" if input_text else instruction
    return f"### Instruction:\n{prompt_instruction}\n\n### Response:\n"


def preprocess_alpaca(examples: Dict[str, List[str]], tokenizer, max_length: int) -> Dict[str, List[List[int]]]:
    """Response-only Alpaca instruction/input/output formatting."""
    if "instruction" not in examples or "output" not in examples:
        raise ValueError("Alpaca preprocessing expects 'instruction' and 'output' columns.")
    instructions = examples["instruction"]
    inputs = examples.get("input", [""] * len(instructions))
    outputs = examples["output"]
    return tokenize_prompt_response_pairs(
        [format_alpaca_prompt(instruction, input_text) for instruction, input_text in zip(instructions, inputs)],
        [(output or "").strip() for output in outputs],
        tokenizer,
        max_length,
    )


def _batch_size(examples: Dict[str, List[object]]) -> int:
    return len(examples[next(iter(examples))]) if examples else 0


def _get_column(
    examples: Dict[str, List[object]], names: List[str], default: Optional[List[object]] = None
) -> List[object]:
    for name in names:
        if name in examples:
            return examples[name]
    if default is not None:
        return default
    raise KeyError(f"Missing required dataset column. Tried: {names}")


def _looks_like_slot_name(text: str) -> bool:
    stripped = text.strip()
    return bool(re.fullmatch(r"[A-Za-z0-9_]+", stripped)) and ("_" in stripped or stripped.islower())


def _parse_massive_slots(utterance: str, annotated_utterance: Optional[object]) -> List[Dict[str, str]]:
    if not isinstance(annotated_utterance, str):
        return []
    slots: List[Dict[str, str]] = []
    for annotation in re.findall(r"\[([^\[\]]+)\]", annotated_utterance):
        if ":" not in annotation:
            continue
        left, right = (part.strip() for part in re.split(r"\s*:\s*", annotation, maxsplit=1))
        if not left or not right:
            continue
        if left in utterance and right not in utterance:
            slot_name, value = right, left
        elif right in utterance and left not in utterance:
            slot_name, value = left, right
        elif _looks_like_slot_name(left) and not _looks_like_slot_name(right):
            slot_name, value = left, right
        elif _looks_like_slot_name(right) and not _looks_like_slot_name(left):
            slot_name, value = right, left
        else:
            slot_name, value = left, right
        slots.append({"slot": slot_name, "value": value})
    return slots


def _intent_to_text(intent_value: object, intent_label_names: Optional[List[str]]) -> str:
    if isinstance(intent_value, str):
        return intent_value
    if isinstance(intent_value, int) and intent_label_names and 0 <= intent_value < len(intent_label_names):
        return intent_label_names[intent_value]
    return "" if intent_value is None else str(intent_value)


def preprocess_massive(
    examples: Dict[str, List[object]], tokenizer, max_length: int, *, intent_label_names: Optional[List[str]] = None
) -> Dict[str, List[List[int]]]:
    """Response-only MASSIVE semantic-parsing formatting."""
    batch_size = _batch_size(examples)
    utterances = _get_column(examples, ["utt", "utterance"])
    intents = _get_column(examples, ["intent_str", "intent_text", "intent"])
    annotated_utterances = _get_column(
        examples, ["annot_utt", "annotated_utt", "annotated_utterance"], [None] * batch_size
    )
    prompts: List[str] = []
    responses: List[str] = []
    for utterance, intent_value, annotated_utterance in zip(utterances, intents, annotated_utterances):
        utterance_text = "" if utterance is None else str(utterance)
        prompts.append(MASSIVE_PROMPT_TEMPLATE.format(utt=utterance_text))
        responses.append(json.dumps({
            "intent": _intent_to_text(intent_value, intent_label_names),
            "slots": _parse_massive_slots(utterance_text, annotated_utterance),
        }, ensure_ascii=False, indent=2))
    return tokenize_prompt_response_pairs(prompts, responses, tokenizer, max_length)


def _resolve_hf_api_bases() -> List[str]:
    endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    return [*([endpoint] if endpoint else []), *([] if endpoint == "https://huggingface.co" else ["https://huggingface.co"])]


def _load_massive_via_parquet_api(dataset_name: str, dataset_config_name: str, dataset_split: str):
    last_error: Optional[Exception] = None
    for api_base in _resolve_hf_api_bases():
        url = f"{api_base}/api/datasets/{dataset_name}/parquet/{dataset_config_name}/{dataset_split}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                parquet_urls = json.loads(response.read().decode("utf-8"))
            if isinstance(parquet_urls, list) and parquet_urls:
                return load_dataset("parquet", data_files={dataset_split: parquet_urls}, split=dataset_split)
            last_error = RuntimeError(f"Parquet API returned no files from {url}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(
        f"Failed to load MASSIVE fallback for {dataset_name}/{dataset_config_name} [{dataset_split}]"
    ) from last_error


def load_training_dataset(args: argparse.Namespace, tokenizer):
    load_kwargs = {"split": args.dataset_split}
    if args.dataset_config_name is not None:
        load_kwargs["name"] = args.dataset_config_name
    try:
        dataset = load_dataset(args.dataset_name, **load_kwargs)
    except RuntimeError as exc:
        if args.dataset != "massive" or "Dataset scripts are no longer supported" not in str(exc):
            raise
        dataset = _load_massive_via_parquet_api(args.dataset_name, args.dataset_config_name, args.dataset_split)
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(len(dataset), args.max_train_samples)))

    if args.dataset == "gsm8k":
        preprocess = lambda batch: preprocess_gsm8k(batch, tokenizer, args.max_seq_length)
    elif args.dataset == "alpaca":
        preprocess = lambda batch: preprocess_alpaca(batch, tokenizer, args.max_seq_length)
    else:
        intent_feature = getattr(dataset, "features", {}).get("intent")
        intent_label_names = list(intent_feature.names) if hasattr(intent_feature, "names") else None
        preprocess = lambda batch: preprocess_massive(
            batch, tokenizer, args.max_seq_length, intent_label_names=intent_label_names
        )
    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {args.dataset} for Gradcodes",
    )
    return dataset, tokenized


def build_infinite_dataloader(
    dataloader: DataLoader,
    *,
    sampler: DistributedSampler | None = None,
    consumed_batches: int = 0,
) -> Iterable[Dict[str, torch.Tensor]]:
    batches_per_epoch = max(1, len(dataloader))
    epoch, batches_to_skip = divmod(max(0, consumed_batches), batches_per_epoch)
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(dataloader):
            if batch_index < batches_to_skip:
                continue
            yield batch
        batches_to_skip = 0
        epoch += 1


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs = {
        "dtype": torch_dtype_from_arg(args.torch_dtype),
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.use_flash_attention:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **load_kwargs,
    )
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        try:
            # Reentrant checkpointing is important here: Gradcodes creates the
            # temporary leaf weights during the recomputation, so each wrapper's
            # last_weight reference points at the tensor that receives a gradient.
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer


def gradcodes_update_l2_penalty(
    modules: List[GradcodesLinear],
    *,
    use_last_weight: bool,
) -> torch.Tensor:
    penalty_terms: List[torch.Tensor] = []

    for module in modules:
        base_codes = module.materialize_base_code_map(dtype=torch.float32)
        base_decoded = module.decode_codes(base_codes).to(torch.float32)

        if use_last_weight and module.last_weight is not None:
            # Keep the search penalty on the detached deployed weight variable
            # while removing any contribution from pure scale drift.
            current_scale_map = module.materialize_scale_map(dtype=torch.float32).detach()
            discrete_update = module.last_weight.to(torch.float32) - (current_scale_map * base_decoded)
        else:
            current_scale_map = module.materialize_scale_map(dtype=torch.float32)
            current_codes = module.current_code()
            current_decoded = module.decode_codes(current_codes).to(torch.float32)
            discrete_update = current_scale_map * (current_decoded - base_decoded)

        penalty_terms.append(discrete_update.pow(2).sum())

    if not penalty_terms:
        return torch.tensor(0.0, dtype=torch.float32)

    return 0.5 * torch.stack(penalty_terms).sum()


def compute_task_objective(
    task_loss: torch.Tensor,
) -> torch.Tensor:
    """
    Shared task-only objective.

    Scale optimization should use this path so any scale weight decay is applied
    only by its optimizer, rather than being injected explicitly into the loss.
    """
    return task_loss.to(torch.float32)


def compute_lattice_search_objective(
    task_loss: torch.Tensor,
    modules: List[GradcodesLinear],
    *,
    lattice_weight_decay: float,
    use_last_weight: bool,
) -> torch.Tensor:
    objective = compute_task_objective(task_loss)
    if lattice_weight_decay > 0.0:
        objective = objective + (
            lattice_weight_decay * gradcodes_update_l2_penalty(modules, use_last_weight=use_last_weight)
        )
    return objective


@torch.no_grad()
def add_lattice_weight_decay_to_captured_gradients(
    modules: List[GradcodesLinear],
    *,
    lattice_weight_decay: float,
) -> torch.Tensor:
    """
    Add the exact L2 penalty gradient without building a dense FP32 graph.

    The legacy objective materialized one FP32 ``discrete_update`` tensor per
    searched weight and retained all of them until backward. For a 3B model that
    alone is roughly 11 GiB. The penalty is quadratic, so its gradient can be
    injected analytically after the task-loss backward pass.
    """
    device = modules[0].storage_device if modules else torch.device("cpu")
    penalty = torch.zeros((), device=device, dtype=torch.float32)
    if lattice_weight_decay <= 0.0:
        return penalty

    for module in modules:
        if module.last_weight is None or module.last_weight.grad is None:
            continue
        base_codes = module.materialize_base_code_map(dtype=torch.float32)
        base_decoded = module.decode_codes(base_codes).to(torch.float32)
        current_scale_map = module.materialize_scale_map(dtype=torch.float32)
        discrete_update = module.last_weight.detach().to(torch.float32) - (
            current_scale_map * base_decoded
        )
        penalty.add_(0.5 * discrete_update.square().sum())
        module.last_weight.grad.add_(
            discrete_update.to(dtype=module.last_weight.grad.dtype),
            alpha=lattice_weight_decay,
        )

    return lattice_weight_decay * penalty


def summarize_lattice_objective_components(
    task_loss: torch.Tensor,
    modules: List[GradcodesLinear],
    *,
    lattice_weight_decay: float,
    use_last_weight: bool,
    device: torch.device,
) -> Dict[str, float]:
    task_objective = compute_task_objective(task_loss)
    lattice_penalty = torch.zeros((), device=task_objective.device, dtype=torch.float32)
    if lattice_weight_decay > 0.0:
        lattice_penalty = lattice_weight_decay * gradcodes_update_l2_penalty(
            modules,
            use_last_weight=use_last_weight,
        )
    lattice_objective = task_objective + lattice_penalty
    task_loss_value, lattice_penalty_value, lattice_objective_value = distributed_mean_tensor_values(
        [task_objective, lattice_penalty, lattice_objective],
        device,
    )
    return {
        "task_loss": task_loss_value,
        "lattice_penalty": lattice_penalty_value,
        "lattice_objective": lattice_objective_value,
    }


def run_forward_backward(
    model,
    batch: Dict[str, torch.Tensor],
    modules: List[GradcodesLinear],
    *,
    lattice_weight_decay: float,
) -> float:
    model.zero_grad(set_to_none=True)
    set_search_gradient_capture(modules, True)
    was_training = model.training
    if getattr(model, "is_gradient_checkpointing", False):
        model.train()
    try:
        outputs = model(**batch)
        task_objective = compute_task_objective(outputs.loss)
        task_objective.backward()
        lattice_penalty = add_lattice_weight_decay_to_captured_gradients(
            modules,
            lattice_weight_decay=lattice_weight_decay,
        )
        objective = task_objective.detach() + lattice_penalty
    finally:
        set_search_gradient_capture(modules, False)
        model.train(was_training)
    sync_gradcodes_gradients(modules)
    return distributed_mean_scalar(objective.detach(), next(model.parameters()).device)


def run_scale_update(
    model,
    batch: Dict[str, torch.Tensor],
    modules: List[GradcodesLinear],
    *,
    scale_optimizer: Optional[torch.optim.Optimizer],
    max_grad_norm: float,
) -> Optional[Dict[str, float]]:
    if scale_optimizer is None:
        return None

    current_lr = float(scale_optimizer.param_groups[0]["lr"])
    if current_lr <= 0.0:
        return None

    scale_parameters = list(iter_scale_parameters(modules))
    if not scale_parameters:
        return None

    model.zero_grad(set_to_none=True)
    scale_optimizer.zero_grad(set_to_none=True)
    set_search_gradient_capture(modules, False)
    was_training = model.training
    if getattr(model, "is_gradient_checkpointing", False):
        model.train()
    try:
        outputs = model(**batch)
        objective = compute_task_objective(outputs.loss)
        objective.backward()
    finally:
        model.train(was_training)
    sync_scale_gradients(modules)
    grad_norm = scale_gradient_norm(scale_parameters)

    if max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(
            scale_parameters,
            max_grad_norm,
        )

    scale_optimizer.step()
    scale_optimizer.zero_grad(set_to_none=True)
    return {
        "task_loss": distributed_mean_scalar(objective.detach(), batch["input_ids"].device),
        "lr": current_lr,
        "grad_norm": grad_norm,
    }


def bootstrap_rank_stage(
    model,
    modules: List[GradcodesLinear],
    batch: Dict[str, torch.Tensor],
    *,
    stage_rank_count: int,
    power_iterations: int,
    lattice_weight_decay: float,
    init_method: str,
) -> float:
    for module in modules:
        module.reset_active_stage()

    bootstrap_loss = run_forward_backward(
        model,
        batch,
        modules,
        lattice_weight_decay=lattice_weight_decay,
    )
    for module in modules:
        if init_method == "lora":
            module.initialize_active_stage_like_lora(
                active_rank_count=stage_rank_count,
            )
        elif init_method == "gradient":
            module.initialize_active_stage_from_gradient(
                active_rank_count=stage_rank_count,
                power_iterations=power_iterations,
            )
        else:
            raise ValueError(f"Unsupported stage initialization method: {init_method}")
    model.zero_grad(set_to_none=True)
    return bootstrap_loss


def snapshot_model_candidate(modules: List[GradcodesLinear]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [module.snapshot_active_stage() for module in modules]


def apply_model_candidate(
    modules: List[GradcodesLinear],
    candidate_state: List[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for module, (a, b) in zip(modules, candidate_state):
        module.set_active_stage(a, b)


def candidate_state_diff_counts(
    candidate_state: List[Tuple[torch.Tensor, torch.Tensor]],
    reference_state: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[int, int]:
    differing_elements = 0
    total_elements = 0
    for (candidate_a, candidate_b), (reference_a, reference_b) in zip(candidate_state, reference_state):
        differing_elements += int(candidate_a.ne(reference_a).sum().item())
        total_elements += int(candidate_a.numel())
        if candidate_b.numel() > 0 or reference_b.numel() > 0:
            differing_elements += int(candidate_b.ne(reference_b).sum().item())
            total_elements += int(candidate_b.numel())

    return differing_elements, total_elements


def build_candidate_batch(
    modules: List[GradcodesLinear],
    args: argparse.Namespace,
    *,
    proposal_lr_a: float,
    proposal_lr_b: float,
) -> Tuple[
    List[List[Tuple[torch.Tensor, torch.Tensor]]],
    List[Dict[str, Union[float, int]]],
    List[Tuple[int, int]],
    Dict[str, Union[float, int]],
]:
    def vector_norm_value(tensor: torch.Tensor) -> float:
        flat = tensor.reshape(-1).to(torch.float32)
        if math.isinf(args.norm_p):
            ord_value: Union[float, int] = float("inf")
        elif float(args.norm_p).is_integer():
            ord_value = int(args.norm_p)
        else:
            ord_value = float(args.norm_p)
        return float(torch.linalg.vector_norm(flat, ord=ord_value).item())

    def aggregate_probability_summaries(probability_tensors: List[torch.Tensor]) -> Dict[str, Union[float, int]]:
        flat = torch.cat([tensor.reshape(-1) for tensor in probability_tensors], dim=0)
        summary = summarize_coordinate_probabilities(flat)
        summary["num_coordinates"] = int(flat.numel())
        summary["coverage"] = float(flat.gt(0).to(torch.float32).mean().item())
        return summary

    def summarize_scaled_guides(
        flat_tensors: List[torch.Tensor],
        pre_clip_norm_values: List[float],
    ) -> Dict[str, Union[float, int]]:
        flat = torch.cat(flat_tensors, dim=0).to(torch.float32)
        flat_abs = flat.abs()
        pre_clip_norms = torch.tensor(pre_clip_norm_values, dtype=torch.float32)
        return {
            "num_entries": int(flat.numel()),
            "min": float(flat.min().item()),
            "abs_mean": float(flat_abs.mean().item()),
            "abs_median": float(flat_abs.median().item()),
            "max": float(flat.max().item()),
            "pre_clip_p_norm_min": float(pre_clip_norms.min().item()),
            "pre_clip_p_norm_mean": float(pre_clip_norms.mean().item()),
            "pre_clip_p_norm_median": float(pre_clip_norms.median().item()),
            "pre_clip_p_norm_max": float(pre_clip_norms.max().item()),
        }

    def summarize_topk_candidate(
        *,
        num_coordinates: int,
        selected_topk_updates: int,
        effective_top_k: int,
    ) -> Dict[str, Union[float, int]]:
        return {
            "joint_log_prob": float("nan"),
            "mean_coordinate_prob": float("nan"),
            "min_coordinate_prob": float("nan"),
            "max_coordinate_prob": float("nan"),
            "num_coordinates": num_coordinates,
            "coverage": float("nan"),
            "selected_topk_updates": selected_topk_updates,
            "effective_top_k": effective_top_k,
            "effective_tau": float("nan"),
        }

    def current_summary_for_tau(
        distribution_geometries: List[Dict[str, Union[bool, torch.Tensor]]],
        tau: float,
    ) -> Dict[str, Union[float, int]]:
        current_probability_tensors: List[torch.Tensor] = []
        for distribution in distribution_geometries:
            if distribution["full_matrix_mode"]:
                code_probs = inverse_distance_probabilities_from_log_distances(
                    distribution["code_log_distances"],
                    tau=tau,
                )
                current_code_probs, _ = candidate_probability_under_distribution(
                    distribution["current_codes"],
                    distribution["code_lattice"],
                    code_probs,
                )
                current_probability_tensors.append(current_code_probs)
            else:
                a_probs = inverse_distance_probabilities_from_log_distances(
                    distribution["a_log_distances"],
                    tau=tau,
                )
                b_probs = inverse_distance_probabilities_from_log_distances(
                    distribution["b_log_distances"],
                    tau=tau,
                )
                current_a_probs, _ = candidate_probability_under_distribution(
                    distribution["current_a"],
                    distribution["a_lattice"],
                    a_probs,
                )
                current_b_probs, _ = candidate_probability_under_distribution(
                    distribution["current_b"],
                    distribution["b_lattice"],
                    b_probs,
                )
                current_probability_tensors.extend([current_a_probs, current_b_probs])

        summary = aggregate_probability_summaries(current_probability_tensors)
        summary["effective_tau"] = float(tau)
        return summary

    def resolve_effective_tau(
        distribution_geometries: List[Dict[str, Union[bool, torch.Tensor]]],
    ) -> Tuple[float, Dict[str, Union[float, int]]]:
        if args.proposal_p <= 0.0:
            fixed_tau = float(args.proposal_tau)
            return fixed_tau, current_summary_for_tau(distribution_geometries, fixed_tau)

        target_log_prob = math.log(args.proposal_p)
        log_tau_min = math.log(1e-6)
        log_tau_max = math.log(1e6)
        coarse_grid = torch.linspace(log_tau_min, log_tau_max, steps=13)

        best_tau = float("nan")
        best_log_tau = float("nan")
        best_gap = float("inf")
        best_summary: Dict[str, Union[float, int]] = {}
        best_index = 0
        evaluations: List[Tuple[float, float, float, Dict[str, Union[float, int]]]] = []

        def is_better_candidate(tau: float, gap: float, current_best_tau: float, current_best_gap: float) -> bool:
            if not math.isfinite(current_best_tau):
                return True
            if gap < current_best_gap - 1e-12:
                return True
            if math.isfinite(gap) and math.isfinite(current_best_gap):
                return abs(gap - current_best_gap) <= 1e-12 and tau < current_best_tau
            if (not math.isfinite(gap)) and (not math.isfinite(current_best_gap)):
                return tau < current_best_tau
            return False

        def evaluate_log_tau(log_tau: float) -> Tuple[float, float, float, Dict[str, Union[float, int]]]:
            tau = float(math.exp(log_tau))
            summary = current_summary_for_tau(distribution_geometries, tau)
            joint_log_prob = float(summary["joint_log_prob"])
            gap = abs(joint_log_prob - target_log_prob) if math.isfinite(joint_log_prob) else float("inf")
            return tau, log_tau, gap, summary

        for index, log_tau_value in enumerate(coarse_grid.tolist()):
            tau, log_tau, gap, summary = evaluate_log_tau(float(log_tau_value))
            evaluations.append((tau, log_tau, gap, summary))
            if is_better_candidate(tau, gap, best_tau, best_gap):
                best_tau = tau
                best_log_tau = log_tau
                best_gap = gap
                best_summary = summary
                best_index = index

        left_log_tau = evaluations[max(0, best_index - 1)][1]
        right_log_tau = evaluations[min(len(evaluations) - 1, best_index + 1)][1]
        if right_log_tau > left_log_tau:
            phi = 0.5 * (1.0 + math.sqrt(5.0))
            a = left_log_tau
            b = right_log_tau
            c = b - ((b - a) / phi)
            d = a + ((b - a) / phi)

            tau_c, log_tau_c, gap_c, summary_c = evaluate_log_tau(c)
            tau_d, log_tau_d, gap_d, summary_d = evaluate_log_tau(d)

            for _ in range(14):
                if gap_c <= gap_d:
                    b = d
                    tau_d, log_tau_d, gap_d, summary_d = tau_c, log_tau_c, gap_c, summary_c
                    d = c
                    c = b - ((b - a) / phi)
                    tau_c, log_tau_c, gap_c, summary_c = evaluate_log_tau(c)
                else:
                    a = c
                    tau_c, log_tau_c, gap_c, summary_c = tau_d, log_tau_d, gap_d, summary_d
                    c = d
                    d = a + ((b - a) / phi)
                    tau_d, log_tau_d, gap_d, summary_d = evaluate_log_tau(d)

            refinement_candidates = [
                (best_tau, best_log_tau, best_gap, best_summary),
                (tau_c, log_tau_c, gap_c, summary_c),
                (tau_d, log_tau_d, gap_d, summary_d),
                evaluate_log_tau(a),
                evaluate_log_tau(b),
                evaluate_log_tau(0.5 * (a + b)),
            ]
            for tau, log_tau, gap, summary in refinement_candidates:
                if is_better_candidate(tau, gap, best_tau, best_gap):
                    best_tau = tau
                    best_log_tau = log_tau
                    best_gap = gap
                    best_summary = summary

        best_summary = dict(best_summary)
        best_summary["adaptive_target_joint_prob"] = float(args.proposal_p)
        best_summary["adaptive_target_joint_log_prob"] = float(target_log_prob)
        best_summary["adaptive_log_prob_gap"] = float(best_gap)
        best_summary["effective_tau"] = float(best_tau)
        return float(best_tau), best_summary

    def materialize_module_distributions(
        distribution_geometries: List[Dict[str, Union[bool, torch.Tensor]]],
        tau: float,
    ) -> List[Dict[str, Union[bool, torch.Tensor]]]:
        module_distributions: List[Dict[str, Union[bool, torch.Tensor]]] = []
        for distribution in distribution_geometries:
            if distribution["full_matrix_mode"]:
                code_probs = inverse_distance_probabilities_from_log_distances(
                    distribution["code_log_distances"],
                    tau=tau,
                )
                module_distributions.append(
                    {
                        "full_matrix_mode": True,
                        "current_codes": distribution["current_codes"],
                        "code_lattice": distribution["code_lattice"],
                        "code_probs": code_probs,
                    }
                )
            else:
                a_probs = inverse_distance_probabilities_from_log_distances(
                    distribution["a_log_distances"],
                    tau=tau,
                )
                b_probs = inverse_distance_probabilities_from_log_distances(
                    distribution["b_log_distances"],
                    tau=tau,
                )
                module_distributions.append(
                    {
                        "full_matrix_mode": False,
                        "current_a": distribution["current_a"],
                        "current_b": distribution["current_b"],
                        "a_lattice": distribution["a_lattice"],
                        "a_probs": a_probs,
                        "b_lattice": distribution["b_lattice"],
                        "b_probs": b_probs,
                    }
                )
        return module_distributions

    distribution_geometries = []
    scaled_guide_tensors: List[torch.Tensor] = []
    pre_clip_norm_values: List[float] = []
    topk_sources: List[Dict[str, int]] = []
    topk_coordinate_offset = 0
    topk_mode = args.proposal_p <= 0.0 and args.proposal_tau == -1.0
    for module_index, module in enumerate(modules):
        current_a, current_b = module.snapshot_active_stage()
        ga, gb = module.proposal_guidance()
        if module.full_matrix_mode:
            raw_step = (proposal_lr_a * ga).to(torch.float32)
            pre_clip_norm_values.append(vector_norm_value(raw_step))
            scaled_step = scaled_proposal_step(
                ga,
                learning_rate=proposal_lr_a,
                min_step_norm=args.min_step_norm if args.min_step_norm > 0.0 else None,
                max_step_norm=args.max_step_norm if args.max_step_norm > 0.0 else None,
                norm_p=args.norm_p,
            )
            scaled_guide_tensors.append(scaled_step.reshape(-1))
            if topk_mode:
                num_codes = int(current_a.numel())
                topk_sources.append(
                    {
                        "module_index": module_index,
                        "tensor_slot": 0,
                        "start": topk_coordinate_offset,
                        "end": topk_coordinate_offset + num_codes,
                    }
                )
                topk_coordinate_offset += num_codes
                continue
            code_lattice, code_log_distances = build_inverse_distance_lattice(
                current_a,
                ga,
                learning_rate=proposal_lr_a,
                max_abs_value=module.elementwise_step_radius,
                epsilon=args.proposal_epsilon,
                min_step_norm=args.min_step_norm if args.min_step_norm > 0.0 else None,
                max_step_norm=args.max_step_norm if args.max_step_norm > 0.0 else None,
                norm_p=args.norm_p,
            )
            distribution_geometries.append(
                {
                    "full_matrix_mode": True,
                    "current_codes": current_a,
                    "code_lattice": code_lattice,
                    "code_log_distances": code_log_distances,
                }
            )
            continue

        raw_step_a = (proposal_lr_a * ga).to(torch.float32)
        raw_step_b = (proposal_lr_b * gb).to(torch.float32)
        pre_clip_norm_values.extend(
            [
                vector_norm_value(raw_step_a),
                vector_norm_value(raw_step_b),
            ]
        )
        scaled_step_a = scaled_proposal_step(
            ga,
            learning_rate=proposal_lr_a,
            min_step_norm=args.min_step_norm if args.min_step_norm > 0.0 else None,
            max_step_norm=args.max_step_norm if args.max_step_norm > 0.0 else None,
            norm_p=args.norm_p,
        )
        scaled_step_b = scaled_proposal_step(
            gb,
            learning_rate=proposal_lr_b,
            min_step_norm=args.min_step_norm if args.min_step_norm > 0.0 else None,
            max_step_norm=args.max_step_norm if args.max_step_norm > 0.0 else None,
            norm_p=args.norm_p,
        )
        scaled_guide_tensors.extend(
            [
                scaled_step_a.reshape(-1),
                scaled_step_b.reshape(-1),
            ]
        )
        if topk_mode:
            num_a = int(current_a.numel())
            num_b = int(current_b.numel())
            topk_sources.append(
                {
                    "module_index": module_index,
                    "tensor_slot": 0,
                    "start": topk_coordinate_offset,
                    "end": topk_coordinate_offset + num_a,
                }
            )
            topk_coordinate_offset += num_a
            topk_sources.append(
                {
                    "module_index": module_index,
                    "tensor_slot": 1,
                    "start": topk_coordinate_offset,
                    "end": topk_coordinate_offset + num_b,
                }
            )
            topk_coordinate_offset += num_b
            continue
        a_lattice, a_log_distances = build_inverse_distance_lattice(
            current_a,
            ga,
            learning_rate=proposal_lr_a,
            max_abs_value=module.amax,
            epsilon=args.proposal_epsilon,
            min_step_norm=args.min_step_norm if args.min_step_norm > 0.0 else None,
            max_step_norm=args.max_step_norm if args.max_step_norm > 0.0 else None,
            norm_p=args.norm_p,
        )
        b_lattice, b_log_distances = build_inverse_distance_lattice(
            current_b,
            gb,
            learning_rate=proposal_lr_b,
            max_abs_value=module.bmax,
            epsilon=args.proposal_epsilon,
            min_step_norm=args.min_step_norm if args.min_step_norm > 0.0 else None,
            max_step_norm=args.max_step_norm if args.max_step_norm > 0.0 else None,
            norm_p=args.norm_p,
        )
        distribution_geometries.append(
            {
                "full_matrix_mode": False,
                "current_a": current_a,
                "current_b": current_b,
                "a_lattice": a_lattice,
                "a_log_distances": a_log_distances,
                "b_lattice": b_lattice,
                "b_log_distances": b_log_distances,
            }
        )

    scaled_guide_summary = summarize_scaled_guides(
        scaled_guide_tensors,
        pre_clip_norm_values,
    )
    if topk_mode:
        base_candidate = snapshot_model_candidate(modules)
        candidates: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [base_candidate]

        all_scaled_steps = torch.cat(scaled_guide_tensors, dim=0).to(torch.float32)
        nonzero_mask = all_scaled_steps.ne(0)
        nonzero_count = int(nonzero_mask.sum().item())
        effective_top_k = min(args.top_k, nonzero_count)
        selected_moves: List[Dict[str, int]] = []

        if effective_top_k > 0:
            masked_scores = all_scaled_steps.abs()
            masked_scores = torch.where(nonzero_mask, masked_scores, torch.full_like(masked_scores, -1.0))
            topk_indices = torch.topk(masked_scores, k=effective_top_k, largest=True, sorted=True).indices.tolist()

            for global_index in topk_indices:
                direction = int(torch.sign(all_scaled_steps[global_index]).item())
                if direction == 0:
                    continue
                for source in topk_sources:
                    if source["start"] <= global_index < source["end"]:
                        selected_moves.append(
                            {
                                "module_index": source["module_index"],
                                "tensor_slot": source["tensor_slot"],
                                "flat_index": global_index - source["start"],
                                "direction": direction,
                            }
                        )
                        break

        effective_top_k = len(selected_moves)
        candidate_probability_summaries = [
            summarize_topk_candidate(
                num_coordinates=int(all_scaled_steps.numel()),
                selected_topk_updates=0,
                effective_top_k=effective_top_k,
            )
        ]

        for mask in range(1, 1 << effective_top_k):
            proposal = [(a.clone(), b.clone()) for a, b in base_candidate]
            selected_update_count = 0
            for bit_index, move in enumerate(selected_moves):
                if not (mask & (1 << bit_index)):
                    continue
                module_index = move["module_index"]
                tensor_slot = move["tensor_slot"]
                flat_index = move["flat_index"]
                direction = move["direction"]
                proposal_a, proposal_b = proposal[module_index]
                if tensor_slot == 0:
                    proposal_a.view(-1)[flat_index] += float(direction)
                else:
                    proposal_b.view(-1)[flat_index] += float(direction)
                proposal[module_index] = (proposal_a, proposal_b)
                selected_update_count += 1
            candidates.append(proposal)
            candidate_probability_summaries.append(
                summarize_topk_candidate(
                    num_coordinates=int(all_scaled_steps.numel()),
                    selected_topk_updates=selected_update_count,
                    effective_top_k=effective_top_k,
                )
            )

        candidate_diff_counts = [
            candidate_state_diff_counts(candidate_state, candidates[0])
            for candidate_state in candidates
        ]

        scaled_guide_summary["effective_tau"] = float("nan")
        scaled_guide_summary["current_joint_log_prob"] = float("nan")
        scaled_guide_summary["current_coverage"] = float("nan")
        scaled_guide_summary["adaptive_target_joint_prob"] = float(args.proposal_p)
        scaled_guide_summary["adaptive_log_prob_gap"] = float("nan")
        return candidates, candidate_probability_summaries, candidate_diff_counts, scaled_guide_summary

    effective_tau, current_candidate_summary = resolve_effective_tau(distribution_geometries)
    module_distributions = materialize_module_distributions(distribution_geometries, effective_tau)

    candidates: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [snapshot_model_candidate(modules)]
    candidate_probability_summaries: List[Dict[str, Union[float, int]]] = [current_candidate_summary]
    scaled_guide_summary["effective_tau"] = float(effective_tau)
    scaled_guide_summary["current_joint_log_prob"] = float(current_candidate_summary["joint_log_prob"])
    scaled_guide_summary["current_coverage"] = float(current_candidate_summary["coverage"])
    scaled_guide_summary["adaptive_target_joint_prob"] = float(args.proposal_p)
    scaled_guide_summary["adaptive_log_prob_gap"] = float(current_candidate_summary.get("adaptive_log_prob_gap", 0.0))

    if args.candidate_batch_size > 1:
        nearest_proposal: List[Tuple[torch.Tensor, torch.Tensor]] = []
        nearest_probability_tensors: List[torch.Tensor] = []
        for distribution in module_distributions:
            if distribution["full_matrix_mode"]:
                nearest_codes, nearest_code_probs, _ = select_nearest_from_inverse_distance_distribution(
                    distribution["code_lattice"],
                    distribution["code_probs"],
                )
                nearest_proposal.append(
                    (
                        nearest_codes,
                        torch.empty(0, device=nearest_codes.device, dtype=torch.float32),
                    )
                )
                nearest_probability_tensors.append(nearest_code_probs)
            else:
                nearest_a, nearest_a_probs, _ = select_nearest_from_inverse_distance_distribution(
                    distribution["a_lattice"],
                    distribution["a_probs"],
                )
                nearest_b, nearest_b_probs, _ = select_nearest_from_inverse_distance_distribution(
                    distribution["b_lattice"],
                    distribution["b_probs"],
                )
                nearest_proposal.append((nearest_a, nearest_b))
                nearest_probability_tensors.extend([nearest_a_probs, nearest_b_probs])
        candidates.append(nearest_proposal)
        candidate_probability_summaries.append(aggregate_probability_summaries(nearest_probability_tensors))

    for _ in range(max(args.candidate_batch_size - 2, 0)):
        proposal: List[Tuple[torch.Tensor, torch.Tensor]] = []
        proposal_probability_tensors: List[torch.Tensor] = []
        for distribution in module_distributions:
            if distribution["full_matrix_mode"]:
                candidate_codes, code_probs, _ = sample_from_inverse_distance_distribution(
                    distribution["code_lattice"],
                    distribution["code_probs"],
                )
                proposal.append(
                    (
                        candidate_codes,
                        torch.empty(0, device=candidate_codes.device, dtype=torch.float32),
                    )
                )
                proposal_probability_tensors.append(code_probs)
            else:
                candidate_a, a_probs, _ = sample_from_inverse_distance_distribution(
                    distribution["a_lattice"],
                    distribution["a_probs"],
                )
                candidate_b, b_probs, _ = sample_from_inverse_distance_distribution(
                    distribution["b_lattice"],
                    distribution["b_probs"],
                )
                proposal.append((candidate_a, candidate_b))
                proposal_probability_tensors.extend([a_probs, b_probs])
        candidates.append(proposal)
        candidate_probability_summaries.append(aggregate_probability_summaries(proposal_probability_tensors))

    candidate_diff_counts = [
        candidate_state_diff_counts(candidate_state, candidates[0])
        for candidate_state in candidates
    ]

    return candidates, candidate_probability_summaries, candidate_diff_counts, scaled_guide_summary


def build_candidate_batch_distributed(
    modules: List[GradcodesLinear],
    args: argparse.Namespace,
    *,
    proposal_lr_a: float,
    proposal_lr_b: float,
) -> Tuple[
    List[List[Tuple[torch.Tensor, torch.Tensor]]],
    List[Dict[str, Union[float, int]]],
    List[Tuple[int, int]],
    Dict[str, Union[float, int]],
]:
    if not distributed_enabled():
        return build_candidate_batch(
            modules,
            args,
            proposal_lr_a=proposal_lr_a,
            proposal_lr_b=proposal_lr_b,
        )

    payload: List[object] = [None]
    if is_main_process():
        candidates, candidate_probability_summaries, candidate_diff_counts, scaled_guide_summary = build_candidate_batch(
            modules,
            args,
            proposal_lr_a=proposal_lr_a,
            proposal_lr_b=proposal_lr_b,
        )
        payload[0] = (
            serialize_candidate_state_for_broadcast(candidates),
            candidate_probability_summaries,
            candidate_diff_counts,
            scaled_guide_summary,
        )

    broadcast_device = modules[0].storage_device if modules else None
    dist.broadcast_object_list(payload, src=0, device=broadcast_device)
    candidates, candidate_probability_summaries, candidate_diff_counts, scaled_guide_summary = payload[0]
    return candidates, candidate_probability_summaries, candidate_diff_counts, scaled_guide_summary


@torch.no_grad()
def evaluate_candidates(
    model,
    batch: Dict[str, torch.Tensor],
    modules: List[GradcodesLinear],
    candidates: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    *,
    lattice_weight_decay: float,
) -> Tuple[Dict[str, float], int, List[Dict[str, float]]]:
    best_summary: Optional[Dict[str, float]] = None
    best_index = 0
    candidate_summaries: List[Dict[str, float]] = []
    current_candidate_summary: Optional[Dict[str, float]] = None
    reference_candidate = candidates[0] if candidates else []

    for idx, candidate in enumerate(candidates):
        if idx > 0:
            differing_elements, _ = candidate_state_diff_counts(candidate, reference_candidate)
            if differing_elements == 0:
                if current_candidate_summary is None:
                    raise RuntimeError("Current candidate loss must be evaluated before reusing it for duplicates.")
                candidate_summaries.append(dict(current_candidate_summary))
                continue

        apply_model_candidate(modules, candidate)
        task_loss = model(**batch).loss
        candidate_summary = summarize_lattice_objective_components(
            task_loss,
            modules,
            lattice_weight_decay=lattice_weight_decay,
            use_last_weight=False,
            device=batch["input_ids"].device,
        )
        if idx == 0:
            current_candidate_summary = dict(candidate_summary)
        candidate_summaries.append(candidate_summary)
        if best_summary is None or candidate_summary["lattice_objective"] < best_summary["lattice_objective"]:
            best_summary = candidate_summary
            best_index = idx

    apply_model_candidate(modules, candidates[best_index])
    if best_summary is None:
        raise RuntimeError("Expected at least one candidate summary.")
    return dict(best_summary), best_index, candidate_summaries


def resolve_stage_rank_schedule(args: argparse.Namespace) -> List[int]:
    if args.ranks_per_stage == -1:
        if args.num_stages is not None:
            if args.num_stages < 1:
                raise ValueError("--num_stages must be at least 1.")
            return [-1] * args.num_stages
        return [-1]

    if args.ranks_per_stage < 1:
        raise ValueError("--ranks_per_stage must be at least 1, or -1 for elementwise grid search.")

    if args.num_stages is not None:
        if args.num_stages < 1:
            raise ValueError("--num_stages must be at least 1.")
        return [args.ranks_per_stage] * args.num_stages

    total_rank_budget = 4 if args.target_rank is None else args.target_rank
    if total_rank_budget < 1:
        raise ValueError("--target_rank must be at least 1 when provided.")

    full_stages, remainder = divmod(total_rank_budget, args.ranks_per_stage)
    schedule = [args.ranks_per_stage] * full_stages
    if remainder:
        schedule.append(remainder)
    return schedule


def format_stage_rank_label(stage_rank: int) -> str:
    return "elem" if stage_rank == -1 else f"{stage_rank:02d}"


def derive_stage_steps(args: argparse.Namespace, dataloader_length: int, num_stages: int) -> int:
    if args.stage_steps is not None:
        return args.stage_steps

    total_budget = max(1, math.ceil(dataloader_length * args.num_train_epochs))
    return max(1, math.ceil(total_budget / num_stages))


def validate_proposal_schedule_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.proposal_lr_warmup_ratio < 1.0:
        raise ValueError("--proposal_lr_warmup_ratio must be in [0, 1).")
    if not 0.0 <= args.proposal_lr_min_ratio <= 1.0:
        raise ValueError("--proposal_lr_min_ratio must be in [0, 1].")
    if args.lattice_weight_decay < 0.0:
        raise ValueError("--lattice_weight_decay/--weight_decay must be non-negative.")
    if args.scale_learning_rate < 0.0:
        raise ValueError("--scale_learning_rate must be non-negative.")
    if args.scale_weight_decay < 0.0:
        raise ValueError("--scale_weight_decay must be non-negative.")
    if not 0.0 <= args.scale_adam_beta1 < 1.0:
        raise ValueError("--scale_adam_beta1 must be in [0, 1).")
    if not 0.0 <= args.scale_adam_beta2 < 1.0:
        raise ValueError("--scale_adam_beta2 must be in [0, 1).")
    if args.scale_adam_epsilon <= 0.0:
        raise ValueError("--scale_adam_epsilon must be positive.")
    if args.scale_max_grad_norm < 0.0:
        raise ValueError("--scale_max_grad_norm must be non-negative.")
    if args.proposal_p < 0.0 or args.proposal_p >= 1.0:
        raise ValueError("--proposal_p must be in [0, 1). Set 0 to disable adaptive tau.")
    if args.proposal_p <= 0.0 and args.proposal_tau != -1.0 and args.proposal_tau <= 0.0:
        raise ValueError("--proposal_tau must be positive, or -1 for deterministic top-k grid search.")
    if args.top_k < 1:
        raise ValueError("--top_k must be at least 1.")
    if args.min_step_norm < 0.0:
        raise ValueError("--min_step_norm must be non-negative.")
    if args.max_step_norm < 0.0:
        raise ValueError("--max_step_norm must be non-negative.")
    if args.norm_p <= 0.0:
        raise ValueError("--norm_p must be positive.")
    if args.min_step_norm > 0.0 and args.max_step_norm > 0.0 and args.min_step_norm > args.max_step_norm:
        raise ValueError("--min_step_norm cannot exceed --max_step_norm.")


def validate_quantization_args(args: argparse.Namespace) -> None:
    if args.quant_bits < 2:
        raise ValueError("--quant_bits must be at least 2.")
    if args.quant_type == "nf4" and args.quant_bits != 4:
        raise ValueError("--quant_type nf4 requires --quant_bits 4.")
    if args.quant_type == "int4" and args.quant_bits != 4:
        raise ValueError("--quant_type int4 requires --quant_bits 4.")
    if args.quant_type == "mxfp4" and args.quant_bits != 4:
        raise ValueError("--quant_type mxfp4 requires --quant_bits 4.")


def resolve_quantization_defaults(args: argparse.Namespace) -> None:
    if args.group_size is None:
        args.group_size = 32 if args.quant_type == "mxfp4" else 64


def proposal_lr_multiplier(
    *,
    step_index: int,
    total_steps: int,
    schedule: str,
    warmup_ratio: float,
    min_ratio: float,
) -> float:
    total_steps = max(1, total_steps)
    current_step = min(max(step_index, 0), total_steps - 1)
    warmup_steps = int(math.floor(total_steps * warmup_ratio))
    if warmup_ratio > 0.0 and warmup_steps == 0:
        warmup_steps = 1

    if warmup_steps > 0 and current_step < warmup_steps:
        return (current_step + 1) / warmup_steps

    if schedule == "constant":
        multiplier = 1.0
    else:
        decay_steps = max(total_steps - warmup_steps - 1, 1)
        progress = (current_step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        if schedule == "linear":
            multiplier = 1.0 - progress
        elif schedule == "cosine":
            multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unsupported proposal LR schedule: {schedule}")

    return max(min_ratio, multiplier)


def get_scheduled_proposal_lrs(
    args: argparse.Namespace,
    *,
    step_index: int,
    total_steps: int,
) -> Tuple[float, float]:
    multiplier = proposal_lr_multiplier(
        step_index=step_index,
        total_steps=total_steps,
        schedule=args.proposal_lr_schedule,
        warmup_ratio=args.proposal_lr_warmup_ratio,
        min_ratio=args.proposal_lr_min_ratio,
    )
    return args.proposal_lr_a * multiplier, args.proposal_lr_b * multiplier


def save_search_checkpoint(
    *,
    output_dir: str,
    tokenizer,
    args: argparse.Namespace,
    metrics: Dict,
    quantized_modules: List[str],
    searchable_modules: List[str],
    model,
    checkpoint_label: str,
    filename: str = "gradcodes_state.pt",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    payload = {
        "base_model_name_or_path": args.model_name_or_path,
        "target_modules": args.target_modules,
        "quantized_modules": quantized_modules,
        "searchable_modules": searchable_modules,
        "args": vars(args),
        "metrics": metrics,
        "checkpoint_label": checkpoint_label,
        "search_state": collect_gradcodes_state(model),
    }
    torch.save(payload, os.path.join(output_dir, filename))

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_model_name_or_path": args.model_name_or_path,
                "quantized_modules": quantized_modules,
                "searchable_modules": searchable_modules,
                "metrics": metrics,
                "checkpoint_label": checkpoint_label,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def write_epoch_wall_clock_times(
    output_dir: str,
    epoch_wall_clock_times: List[Dict[str, Union[float, int]]],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, EPOCH_WALL_CLOCK_TIMES_FILENAME)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "unit": "seconds",
                "epochs": epoch_wall_clock_times,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    os.replace(tmp_path, path)
    return path


def resolve_resume_checkpoint_path(path: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(expanded):
        expanded = os.path.join(expanded, "gradcodes_state.pt")
    if not os.path.isfile(expanded):
        raise FileNotFoundError(f"Resume checkpoint not found: {expanded}")
    return expanded


def load_resume_payload(path: str) -> Tuple[str, Dict]:
    checkpoint_path = resolve_resume_checkpoint_path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_keys = {"base_model_name_or_path", "search_state", "args", "metrics"}
    missing = sorted(required_keys - set(payload))
    if missing:
        raise ValueError(f"Resume checkpoint is missing required fields: {missing}")
    return checkpoint_path, payload


def infer_resume_progress(
    payload: Dict,
    *,
    epoch_num_batches: int,
    target_stage_steps: int,
    target_num_stages: int,
) -> Dict[str, Union[int, float, List[Dict[str, Union[float, int]]]]]:
    checkpoint_metrics = payload.get("metrics", {})
    epoch_times = list(checkpoint_metrics.get("epoch_wall_clock_times", []))
    latest_epoch = epoch_times[-1] if epoch_times else {}
    global_step = int(checkpoint_metrics.get("global_steps", latest_epoch.get("global_step", 0)))
    consumed_batches = int(
        checkpoint_metrics.get("consumed_batches", latest_epoch.get("consumed_batches", global_step + 1))
    )
    saved_epoch_count = int(latest_epoch.get("epoch", consumed_batches // max(1, epoch_num_batches)))

    if epoch_times:
        saved_batches_per_epoch = int(epoch_times[0]["consumed_batches"]) // max(1, int(epoch_times[0]["epoch"]))
        if saved_batches_per_epoch != epoch_num_batches:
            raise ValueError(
                "Resume dataloader length does not match the checkpoint: "
                f"current={epoch_num_batches}, checkpoint={saved_batches_per_epoch}. "
                "Use the same dataset size, world size, and per-device batch size as the original run."
            )

    completed_stages = len(checkpoint_metrics.get("stages", []))
    active_ranks = {
        int(module_state.get("active_stage_rank_count", 0))
        for module_state in payload["search_state"].values()
        if bool(int(module_state.get("search_enabled", 0)))
    }
    if len(active_ranks) != 1:
        raise ValueError(f"Checkpoint has inconsistent active stage ranks: {sorted(active_ranks)}")
    active_rank = next(iter(active_ranks), 0)

    checkpoint_args = payload.get("args", {})
    original_num_stages = int(checkpoint_args.get("num_stages") or 1)
    original_stage_steps_arg = checkpoint_args.get("stage_steps")
    if original_stage_steps_arg is not None:
        original_stage_steps = int(original_stage_steps_arg)
    else:
        original_epoch_budget = max(
            1,
            math.ceil(epoch_num_batches * float(checkpoint_args.get("num_train_epochs", 1.0))),
        )
        original_stage_steps = max(1, math.ceil(original_epoch_budget / original_num_stages))

    resume_stage_index = completed_stages
    if resume_stage_index > target_num_stages:
        raise ValueError(
            f"Checkpoint already completed {resume_stage_index} stages, but the current run has only "
            f"{target_num_stages}."
        )
    if active_rank > 0:
        if resume_stage_index >= target_num_stages:
            raise ValueError("Checkpoint has an active stage that is absent from the current stage schedule.")
        stage_step = global_step - (completed_stages * original_stage_steps)
        if stage_step < 0:
            raise ValueError("Could not infer a valid stage step from the checkpoint metrics.")
        if stage_step > target_stage_steps:
            raise ValueError(
                f"Checkpoint is at stage step {stage_step}, beyond the current target of {target_stage_steps}. "
                "Increase --num_train_epochs or --stage_steps."
            )
    else:
        stage_step = 0

    previous_elapsed = float(checkpoint_metrics.get("elapsed_seconds", 0.0))
    if previous_elapsed <= 0.0:
        previous_elapsed = sum(float(entry.get("wall_clock_seconds", 0.0)) for entry in epoch_times)

    return {
        "global_step": global_step,
        "consumed_batches": consumed_batches,
        "saved_epoch_count": saved_epoch_count,
        "resume_stage_index": resume_stage_index,
        "resume_stage_step": stage_step,
        "active_rank": active_rank,
        "previous_elapsed": previous_elapsed,
        "epoch_wall_clock_times": epoch_times,
    }


def main(default_dataset: str = "gsm8k") -> None:
    args = parse_args(default_dataset=default_dataset)
    resolve_dataset_defaults(args)
    resolve_quantization_defaults(args)
    validate_proposal_schedule_args(args)
    validate_quantization_args(args)
    resume_checkpoint_path: Optional[str] = None
    resume_payload: Optional[Dict] = None
    if args.resume_from_checkpoint is not None:
        resume_checkpoint_path, resume_payload = load_resume_payload(args.resume_from_checkpoint)
        saved_base_model = str(resume_payload["base_model_name_or_path"])
        if saved_base_model != args.model_name_or_path:
            raise ValueError(
                f"Resume checkpoint uses base model {saved_base_model!r}, but --model_name_or_path is "
                f"{args.model_name_or_path!r}."
            )
    world_size, local_rank = initialize_distributed()
    seed_everything(args.seed)

    use_cuda = cuda_available()
    if use_cuda:
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model, tokenizer = load_model_and_tokenizer(args)

    quantized_modules, searchable_modules = replace_linear_with_gradcodes(
        model,
        quantized_modules=args.quantized_modules,
        target_modules=args.target_modules,
        bits=args.quant_bits,
        group_size=args.group_size,
        quant_type=args.quant_type,
        stage_rank=args.ranks_per_stage,
        amax=args.amax,
        bmax=args.bmax,
        capture_weight_dtype=(
            None
            if args.gradient_capture_dtype == "model"
            else torch_dtype_from_arg(args.gradient_capture_dtype)
        ),
    )
    if not quantized_modules:
        raise ValueError("No linear modules were quantized. Check --quantized_modules.")
    if not searchable_modules:
        raise ValueError("No linear modules were marked searchable. Check --target_modules.")

    if resume_payload is not None:
        loaded_modules = load_gradcodes_state(model, resume_payload["search_state"], strict=True)
        if len(loaded_modules) != len(quantized_modules):
            raise RuntimeError(
                f"Loaded {len(loaded_modules)} Gradcodes modules, expected {len(quantized_modules)}."
            )

    model.to(device)
    model.eval()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    raw_dataset, train_dataset = load_training_dataset(args, tokenizer)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=distributed_rank(),
            shuffle=True,
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if use_cuda else None,
            return_tensors="pt",
        ),
        num_workers=args.dataloader_num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    modules = [module for _, module in iter_gradcodes_modules(model, search_only=True)]
    scale_optimizer = build_scale_optimizer(modules, args)
    scale_parameter_count = count_scale_parameters(modules)
    stage_rank_schedule = resolve_stage_rank_schedule(args)
    num_stages = len(stage_rank_schedule)
    elementwise_stage_mode = any(stage_rank == -1 for stage_rank in stage_rank_schedule)
    total_rank_budget = None if elementwise_stage_mode else sum(stage_rank_schedule)
    stage_steps = derive_stage_steps(args, len(train_dataloader), num_stages)
    total_search_steps = max(1, num_stages * stage_steps)
    epoch_num_batches = max(1, len(train_dataloader))
    topk_mode_active = args.proposal_p <= 0.0 and args.proposal_tau == -1.0
    resume_progress: Optional[Dict[str, Union[int, float, List[Dict[str, Union[float, int]]]]]] = None
    if resume_payload is not None:
        resume_progress = infer_resume_progress(
            resume_payload,
            epoch_num_batches=epoch_num_batches,
            target_stage_steps=stage_steps,
            target_num_stages=num_stages,
        )
        active_rank = int(resume_progress["active_rank"])
        resume_stage_index = int(resume_progress["resume_stage_index"])
        if active_rank > 0 and active_rank != stage_rank_schedule[resume_stage_index]:
            raise ValueError(
                f"Checkpoint active rank is {active_rank}, but current stage {resume_stage_index + 1} "
                f"expects rank {stage_rank_schedule[resume_stage_index]}."
            )

    initial_consumed_batches = 0 if resume_progress is None else int(resume_progress["consumed_batches"])
    batch_iterator = build_infinite_dataloader(
        train_dataloader,
        sampler=train_sampler,
        consumed_batches=initial_consumed_batches,
    )

    if is_main_process():
        print("=" * 80)
        print("Gradcodes Fine-tuning on GSM8K")
        print("=" * 80)
        print(f"Model: {args.model_name_or_path}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Dataset: {args.dataset_name}/{args.dataset_config_name} [{args.dataset_split}]")
        print(f"Raw dataset size: {len(raw_dataset)}")
        print(f"Quantized modules: {len(quantized_modules)}")
        print(f"Searchable modules: {len(searchable_modules)}")
        print(f"Number of stages: {num_stages}")
        if elementwise_stage_mode:
            print("Ranks per stage (max): -1 (elementwise grid mode)")
            print("Total low-rank budget: n/a (elementwise grid mode)")
        else:
            print(f"Ranks per stage (max): {args.ranks_per_stage}")
            print(f"Total low-rank budget: {total_rank_budget}")
        print(f"Stage rank schedule: {stage_rank_schedule}")
        print(f"Stage steps: {stage_steps}")
        print(f"Total inner search steps: {total_search_steps}")
        if topk_mode_active:
            print(
                f"Candidate batch size: ignored in deterministic top-k mode "
                f"(top_k={args.top_k}, candidates_per_step={1 << args.top_k})"
            )
        else:
            print(f"Candidate batch size: {args.candidate_batch_size}")
        print(f"Stage init method: {args.stage_init_method}")
        print(
            "Proposal LR schedule: "
            f"{args.proposal_lr_schedule} | "
            f"base_lr_a={args.proposal_lr_a:.6e} | "
            f"base_lr_b={args.proposal_lr_b:.6e} | "
            f"warmup_ratio={args.proposal_lr_warmup_ratio:.4f} | "
            f"min_ratio={args.proposal_lr_min_ratio:.4f}"
        )
        if args.proposal_p > 0.0:
            print(
                f"Adaptive current joint probability target: {args.proposal_p:.6e} "
                f"(overrides proposal_tau each step)"
            )
            if args.proposal_tau == -1.0:
                print("Proposal tau: overridden by adaptive p (deterministic top-k mode disabled)")
            else:
                print(f"Base proposal tau: {args.proposal_tau:.6f} (overridden by adaptive p)")
        elif args.proposal_tau == -1.0:
            print(f"Proposal tau: {args.proposal_tau:.6f} (deterministic top-k one-step grid search)")
        else:
            print(f"Proposal tau: {args.proposal_tau:.6f}")
        if elementwise_stage_mode:
            print("Elementwise stage mode: proposal_lr_a drives grid updates and proposal_lr_b is ignored.")
        if args.min_step_norm > 0.0 or args.max_step_norm > 0.0:
            print(
                f"Proposal step norm bounds: min={args.min_step_norm:.6e} | "
                f"max={args.max_step_norm:.6e} | p={args.norm_p}"
            )
        else:
            print("Proposal step norm bounds: disabled")
        print(f"Lattice search-objective weight decay: {args.lattice_weight_decay:.6e}")
        if scale_optimizer is None:
            print("Scale AdamW: disabled")
        else:
            print(
                "Scale AdamW: "
                f"lr={args.scale_learning_rate:.6e} | "
                f"weight_decay={args.scale_weight_decay:.6e} | "
                f"betas=({args.scale_adam_beta1:.4f}, {args.scale_adam_beta2:.4f}) | "
                f"eps={args.scale_adam_epsilon:.2e} | "
                f"max_grad_norm={args.scale_max_grad_norm:.6e}"
            )
            print(f"Learnable block-shared scale parameters: {scale_parameter_count}")
        print(f"Quantization type/bits/group size: {args.quant_type}/{args.quant_bits}/{args.group_size}")
        print(f"Gradient capture dtype: {args.gradient_capture_dtype}")
        print(f"Gradient checkpointing: {args.gradient_checkpointing}")
        print("Batch padding: dynamic (pad-to-multiple-of-8 on CUDA)")
        if resume_progress is not None:
            print(f"Resume checkpoint: {resume_checkpoint_path}")
            print(
                "Resume progress: "
                f"global_step={int(resume_progress['global_step'])} | "
                f"consumed_batches={int(resume_progress['consumed_batches'])} | "
                f"stage={int(resume_progress['resume_stage_index']) + 1} | "
                f"stage_step={int(resume_progress['resume_stage_step'])}"
            )
            print("Resume optimizer/RNG: unavailable in this legacy checkpoint; AdamW and RNG restart.")
        print(f"Quantized module patterns: {args.quantized_modules}")
        print(f"Search target patterns: {args.target_modules}")
        print(f"Local proposal half-width: a +/- {args.amax}, b +/- {args.bmax}")
        print("-" * 80)
        print("[quantized]")
        for name in quantized_modules:
            print(f"  - {name}")
        print("[searchable]")
        for name in searchable_modules:
            print(f"  - {name}")
        print("=" * 80)

    checkpoint_metrics = {} if resume_payload is None else resume_payload.get("metrics", {})
    metrics: Dict = {
        "stages": list(checkpoint_metrics.get("stages", [])),
    }
    epoch_wall_clock_times: List[Dict[str, Union[float, int]]] = (
        []
        if resume_progress is None
        else list(resume_progress["epoch_wall_clock_times"])
    )
    metrics["epoch_wall_clock_times"] = epoch_wall_clock_times
    global_step = 0 if resume_progress is None else int(resume_progress["global_step"])
    total_accepted_steps = sum(int(stage.get("accepted_steps", 0)) for stage in metrics["stages"])
    consumed_batches = 0 if resume_progress is None else int(resume_progress["consumed_batches"])
    saved_epoch_count = 0 if resume_progress is None else int(resume_progress["saved_epoch_count"])
    total_scale_updates = int(checkpoint_metrics.get("scale_updates", global_step if resume_progress else 0))
    previous_elapsed = 0.0 if resume_progress is None else float(resume_progress["previous_elapsed"])
    synchronize_wall_clock_timer(device)
    start_time = time.time()
    epoch_start_time = time.perf_counter()

    def save_epoch_checkpoints_if_needed() -> None:
        nonlocal saved_epoch_count, epoch_start_time
        completed_epochs = consumed_batches // epoch_num_batches
        while saved_epoch_count < completed_epochs:
            epoch_number = saved_epoch_count + 1
            synchronize_wall_clock_timer(device)
            epoch_end_time = time.perf_counter()
            epoch_wall_clock_seconds = distributed_max_scalar(epoch_end_time - epoch_start_time, device)
            if is_main_process():
                epoch_wall_clock_record = {
                    "epoch": epoch_number,
                    "wall_clock_seconds": round(epoch_wall_clock_seconds, 3),
                    "consumed_batches": epoch_number * epoch_num_batches,
                    "global_step": global_step,
                }
                epoch_wall_clock_times.append(epoch_wall_clock_record)
                metrics["epoch_wall_clock_times"] = epoch_wall_clock_times
                metrics["global_steps"] = global_step
                metrics["consumed_batches"] = consumed_batches
                metrics["saved_epoch_checkpoints"] = epoch_number
                if scale_optimizer is not None:
                    metrics["scale_updates"] = total_scale_updates
                epoch_times_path = write_epoch_wall_clock_times(args.output_dir, epoch_wall_clock_times)
                print(
                    f"[time] epoch {epoch_number} wall_clock_seconds="
                    f"{epoch_wall_clock_seconds:.3f} | wrote {epoch_times_path}"
                )

                epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch_number:04d}")
                save_search_checkpoint(
                    output_dir=epoch_output_dir,
                    tokenizer=tokenizer,
                    args=args,
                    metrics=metrics,
                    quantized_modules=quantized_modules,
                    searchable_modules=searchable_modules,
                    model=model,
                    checkpoint_label=f"epoch_{epoch_number:04d}",
                )
                print(f"[save] wrote search-state checkpoint for epoch {epoch_number} to {epoch_output_dir}")
            saved_epoch_count = epoch_number
            if distributed_enabled():
                dist.barrier()
            synchronize_wall_clock_timer(device)
            epoch_start_time = time.perf_counter()

    for stage_index, stage_rank_count in enumerate(stage_rank_schedule):
        stage_idx = stage_index + 1
        if resume_progress is not None and stage_index < int(resume_progress["resume_stage_index"]):
            continue

        resuming_active_stage = bool(
            resume_progress is not None
            and stage_index == int(resume_progress["resume_stage_index"])
            and int(resume_progress["active_rank"]) > 0
        )
        if resuming_active_stage:
            bootstrap_loss = float("nan")
            first_stage_step = int(resume_progress["resume_stage_step"]) + 1
            if is_main_process():
                print(
                    f"[resume] continuing stage {stage_idx} at step "
                    f"{first_stage_step}/{stage_steps}; bootstrap is not repeated."
                )
        else:
            init_batch = move_batch_to_device(next(batch_iterator), device)
            consumed_batches += 1
            bootstrap_loss = bootstrap_rank_stage(
                model,
                modules,
                init_batch,
                stage_rank_count=stage_rank_count,
                power_iterations=args.power_iterations,
                lattice_weight_decay=args.lattice_weight_decay,
                init_method=args.stage_init_method,
            )
            synchronize_active_stage_from_main(modules)
            save_epoch_checkpoints_if_needed()
            first_stage_step = 1

        accepted_steps = 0
        last_selected_loss = bootstrap_loss
        last_scale_loss: Optional[float] = None
        stage_scale_updates = 0

        for stage_step in range(first_stage_step, stage_steps + 1):
            proposal_lr_a, proposal_lr_b = get_scheduled_proposal_lrs(
                args,
                step_index=global_step,
                total_steps=total_search_steps,
            )
            batch = move_batch_to_device(next(batch_iterator), device)
            consumed_batches += 1
            search_skipped = should_skip_search_update(
                modules,
                proposal_lr_a=proposal_lr_a,
                proposal_lr_b=proposal_lr_b,
            )
            best_candidate_summary: Optional[Dict[str, float]] = None
            best_index = 0
            candidate_summaries: List[Dict[str, float]] = []
            candidate_probability_summaries: List[Dict[str, Union[float, int]]] = []
            candidate_diff_counts: List[Tuple[int, int]] = []
            scaled_guide_summary: Optional[Dict[str, Union[float, int]]] = None

            if not search_skipped:
                run_forward_backward(
                    model,
                    batch,
                    modules,
                    lattice_weight_decay=args.lattice_weight_decay,
                )
                candidates, candidate_probability_summaries, candidate_diff_counts, scaled_guide_summary = build_candidate_batch_distributed(
                    modules,
                    args,
                    proposal_lr_a=proposal_lr_a,
                    proposal_lr_b=proposal_lr_b,
                )
                best_candidate_summary, best_index, candidate_summaries = evaluate_candidates(
                    model,
                    batch,
                    modules,
                    candidates,
                    lattice_weight_decay=args.lattice_weight_decay,
                )
                model.zero_grad(set_to_none=True)

            scale_update_info = run_scale_update(
                model,
                batch,
                modules,
                scale_optimizer=scale_optimizer,
                max_grad_norm=args.scale_max_grad_norm,
            )
            if scale_update_info is not None:
                last_scale_loss = scale_update_info["task_loss"]
                stage_scale_updates += 1
                total_scale_updates += 1

            if (not search_skipped) and best_index != 0:
                accepted_steps += 1
                total_accepted_steps += 1
            if best_candidate_summary is not None:
                last_selected_loss = best_candidate_summary["lattice_objective"]
            elif scale_update_info is not None:
                last_selected_loss = scale_update_info["task_loss"]
            global_step += 1

            if is_main_process():
                summary_parts = [
                    f"[stage {stage_idx:02d}/{num_stages:02d}]",
                    f"step {stage_step:04d}/{stage_steps:04d}",
                    f"stage_rank={format_stage_rank_label(stage_rank_count)}",
                    f"lr_a={proposal_lr_a:.6e}",
                    f"lr_b={proposal_lr_b:.6e}",
                ]
                if search_skipped:
                    summary_parts.append("search=skipped(lr=0)")
                else:
                    effective_tau = float(scaled_guide_summary["effective_tau"])
                    tau_label = "n/a" if not math.isfinite(effective_tau) else f"{effective_tau:.6e}"
                    summary_parts.extend(
                        [
                            f"tau={tau_label}",
                            f"accepted={'yes' if best_index != 0 else 'no'}",
                        ]
                    )
                summary_parts.append(f"total_accepted_steps={total_accepted_steps}")
                print(" | ".join(summary_parts))
                if not search_skipped:
                    for candidate_index, (candidate_summary, probability_summary, candidate_diff_count) in enumerate(
                        zip(candidate_summaries, candidate_probability_summaries, candidate_diff_counts)
                    ):
                        if topk_mode_active:
                            candidate_kind = "current" if candidate_index == 0 else "topk_subset"
                        elif candidate_index == 0:
                            candidate_kind = "current"
                        elif candidate_index == 1:
                            candidate_kind = "nearest"
                        else:
                            candidate_kind = "sampled"
                        print(
                            f"  cand[{candidate_index:02d}] "
                            f"kind={candidate_kind} | "
                            f"task_loss={candidate_summary['task_loss']:.6f} | "
                            f"diff_vs_current={candidate_diff_count[0]}/{candidate_diff_count[1]} | "
                            f"lattice_penalty={candidate_summary['lattice_penalty']:.6f} | "
                            f"lattice_objective={candidate_summary['lattice_objective']:.6f} | "
                            f"joint_log_prob={probability_summary['joint_log_prob']:.6f}"
                            f"{' | selected' if candidate_index == best_index else ''}"
                        )
                    if scaled_guide_summary is not None and args.proposal_p > 0.0:
                        print(
                            f"  adaptive_tau "
                            f"target_current_joint_prob={args.proposal_p:.6e} | "
                            f"current_joint_log_prob={float(scaled_guide_summary['current_joint_log_prob']):.6f} | "
                            f"current_coverage={float(scaled_guide_summary['current_coverage']):.6f} | "
                            f"log_prob_gap={float(scaled_guide_summary['adaptive_log_prob_gap']):.6f}"
                        )
                    if scaled_guide_summary is not None:
                        print(
                            f"  lr_times_guide_overall "
                            f"num_entries={scaled_guide_summary['num_entries']} | "
                            f"p={args.norm_p} | "
                            f"min={scaled_guide_summary['min']:.6f} | "
                            f"abs_mean={scaled_guide_summary['abs_mean']:.6f} | "
                            f"abs_median={scaled_guide_summary['abs_median']:.6f} | "
                            f"max={scaled_guide_summary['max']:.6f} | "
                            f"pre_clip_p_norm_min={scaled_guide_summary['pre_clip_p_norm_min']:.6f} | "
                            f"pre_clip_p_norm_mean={scaled_guide_summary['pre_clip_p_norm_mean']:.6f} | "
                            f"pre_clip_p_norm_median={scaled_guide_summary['pre_clip_p_norm_median']:.6f} | "
                            f"pre_clip_p_norm_max={scaled_guide_summary['pre_clip_p_norm_max']:.6f}"
                        )
                if scale_update_info is not None:
                    print(
                        f"  [scale] "
                        f"task_loss={scale_update_info['task_loss']:.6f} | "
                        f"lr={scale_update_info['lr']:.6e} | "
                        f"grad_norm={scale_update_info['grad_norm']:.6f}"
                    )
            save_epoch_checkpoints_if_needed()

        for module in modules:
            module.commit_active_stage()

        stage_summary = {
            "stage": stage_idx,
            "stage_rank": stage_rank_count,
            "bootstrap_loss": None if resuming_active_stage else round(bootstrap_loss, 6),
            "final_selected_loss": round(last_selected_loss, 6),
            "final_selected_lattice_objective": round(last_selected_loss, 6),
            "accepted_steps": accepted_steps,
            "stage_steps": stage_steps,
        }
        if resuming_active_stage:
            stage_summary["resumed_from_stage_step"] = first_stage_step - 1
            stage_summary["accepted_steps_since_resume"] = accepted_steps
        if last_scale_loss is not None:
            stage_summary["final_scale_loss"] = round(last_scale_loss, 6)
            stage_summary["final_scale_task_loss"] = round(last_scale_loss, 6)
            stage_summary["scale_updates"] = stage_scale_updates
        metrics["stages"].append(stage_summary)
        if is_main_process():
            bootstrap_label = "resumed" if resuming_active_stage else f"{bootstrap_loss:.4f}"
            print(
                f"[stage {stage_idx:02d}] committed stage block | "
                f"stage_rank={stage_rank_count} | "
                f"bootstrap_lattice_objective={bootstrap_label} | "
                f"final_selected_lattice_objective={last_selected_loss:.4f} | "
                f"accepted_steps={accepted_steps}/{stage_steps}"
                f"{'' if last_scale_loss is None else f' | final_scale_task_loss={last_scale_loss:.4f} | scale_updates={stage_scale_updates}'}"
            )

    resumed_elapsed = time.time() - start_time
    elapsed = previous_elapsed + resumed_elapsed
    metrics["elapsed_seconds"] = round(elapsed, 3)
    metrics["global_steps"] = global_step
    metrics["consumed_batches"] = consumed_batches
    metrics["saved_epoch_checkpoints"] = saved_epoch_count
    if scale_optimizer is not None:
        metrics["scale_updates"] = total_scale_updates
    if use_cuda:
        synchronize_wall_clock_timer(device)
        metrics["cuda_memory_gib"] = {
            "allocated": round(torch.cuda.memory_allocated(device) / (1024**3), 3),
            "reserved": round(torch.cuda.memory_reserved(device) / (1024**3), 3),
            "peak_allocated": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
            "peak_reserved": round(torch.cuda.max_memory_reserved(device) / (1024**3), 3),
        }

    if is_main_process():
        epoch_times_path = write_epoch_wall_clock_times(args.output_dir, epoch_wall_clock_times)
        save_search_checkpoint(
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            args=args,
            metrics=metrics,
            quantized_modules=quantized_modules,
            searchable_modules=searchable_modules,
            model=model,
            checkpoint_label="final",
        )

        print("=" * 80)
        print("Gradcodes training finished.")
        print(f"Elapsed: {elapsed:.2f}s total ({resumed_elapsed:.2f}s this process)")
        if use_cuda:
            cuda_memory = metrics["cuda_memory_gib"]
            print(
                "CUDA memory (GiB): "
                f"allocated={cuda_memory['allocated']:.3f} | "
                f"reserved={cuda_memory['reserved']:.3f} | "
                f"peak_allocated={cuda_memory['peak_allocated']:.3f} | "
                f"peak_reserved={cuda_memory['peak_reserved']:.3f}"
            )
        print(f"Epoch wall-clock times: {epoch_times_path}")
        print(f"Artifacts: {args.output_dir}")
        print("=" * 80)

    finalize_distributed()


if __name__ == "__main__":
    main()
