"""
QLoRA fine-tuning for GSM8K, Alpaca, and MASSIVE.

Usage:
    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_qlora.py --dataset gsm8k
"""
import os
import json
import argparse
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Union

import torch

# Set Hugging Face mirror for faster downloads in China
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Set CUDA visible devices (modify as needed)
#os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import transformers
from configs import (
    ModelArguments,
    DataArguments,
    LoraArguments,
    TrainingArguments as TrainArgs,
    QuantizationArguments,
)


EPOCH_WALL_CLOCK_TIMES_FILENAME = "epoch_wall_clock_times.json"
MASSIVE_PROMPT_TEMPLATE = """You are a mobile voice assistant semantic parser.
Extract the user's intent and slots from the utterance.

Output JSON only in the format:
{{
  "intent": "...",
  "slots": [{{"slot": "...", "value": "..."}}]
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
        "output_dir": "./outputs/llama-3.2-3b-qlora-gsm8k",
    },
    "alpaca": {
        "dataset_name": "yahma/alpaca-cleaned",
        "dataset_config_name": None,
        "dataset_split": "train",
        "output_dir": "./outputs/llama-3.2-3b-qlora-alpaca",
    },
    "massive": {
        "dataset_name": "AmazonScience/massive",
        "dataset_config_name": "en-US",
        "dataset_split": "train",
        "output_dir": "./outputs/llama-3.2-3b-qlora-massive",
    },
}


def distributed_enabled() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def wall_clock_timer_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if 0 <= local_rank < torch.cuda.device_count():
            return torch.device("cuda", local_rank)
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def synchronize_wall_clock_timer() -> torch.device:
    device = wall_clock_timer_device()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return device


def distributed_max_scalar(value: Union[float, torch.Tensor]) -> float:
    device = wall_clock_timer_device()
    tensor = value.detach().to(device=device, dtype=torch.float32) if isinstance(value, torch.Tensor) else torch.tensor(
        float(value),
        device=device,
        dtype=torch.float32,
    )
    if distributed_enabled():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


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


def is_world_process_zero(args, state) -> bool:
    if hasattr(state, "is_world_process_zero"):
        return bool(state.is_world_process_zero)
    if hasattr(args, "process_index"):
        return int(args.process_index) == 0
    return int(os.environ.get("RANK", "0")) == 0


class EpochWallClockTimerCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.epoch_wall_clock_times: List[Dict[str, Union[float, int]]] = []
        self.epoch_start_time = None
        self.pending_epoch_timer_reset = False

    def _reset_epoch_timer(self, *, barrier: bool = False) -> None:
        if barrier and distributed_enabled():
            torch.distributed.barrier()
        synchronize_wall_clock_timer()
        self.epoch_start_time = time.perf_counter()
        self.pending_epoch_timer_reset = False

    def on_train_begin(self, args, state, control, **kwargs):
        self._reset_epoch_timer()

    def on_step_begin(self, args, state, control, **kwargs):
        if self.pending_epoch_timer_reset:
            self._reset_epoch_timer(barrier=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is None:
            self._reset_epoch_timer()
            return

        synchronize_wall_clock_timer()
        epoch_end_time = time.perf_counter()
        epoch_wall_clock_seconds = distributed_max_scalar(epoch_end_time - self.epoch_start_time)
        epoch_number = len(self.epoch_wall_clock_times) + 1

        if is_world_process_zero(args, state):
            epoch_wall_clock_record = {
                "epoch": epoch_number,
                "wall_clock_seconds": round(epoch_wall_clock_seconds, 3),
                "global_step": int(state.global_step),
            }
            if state.epoch is not None:
                epoch_wall_clock_record["trainer_epoch"] = round(float(state.epoch), 6)
            self.epoch_wall_clock_times.append(epoch_wall_clock_record)
            epoch_times_path = write_epoch_wall_clock_times(self.output_dir, self.epoch_wall_clock_times)
            print(
                f"[time] epoch {epoch_number} wall_clock_seconds="
                f"{epoch_wall_clock_seconds:.3f} | wrote {epoch_times_path}"
            )

        if distributed_enabled():
            torch.distributed.barrier()
        self.pending_epoch_timer_reset = True

    def write(self) -> str:
        return write_epoch_wall_clock_times(self.output_dir, self.epoch_wall_clock_times)


def load_model_and_tokenizer(
    model_args: ModelArguments,
    quant_args: QuantizationArguments,
    train_args: TrainArgs,
    local_rank: int = 0,
):
    """Load model and tokenizer with 4-bit quantization"""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_args.load_in_4bit,
        bnb_4bit_quant_type=quant_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quant_args.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=quant_args.bnb_4bit_use_double_quant,
    )

    # Get current device index for multi-GPU training
    device_index = local_rank

    # Load model with quantization
    load_kwargs = {
        "quantization_config": bnb_config,
        "device_map": {"": device_index},  # Place model on current GPU
        "trust_remote_code": True,
    }

    # Only add flash_attention if supported (transformers >= 4.35.0)
    if model_args.use_flash_attention:
        try:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except TypeError:
            pass  # Not supported in this version

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **load_kwargs
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def apply_lora(
    model,
    lora_args: LoraArguments,
):
    """Apply LoRA adapters to the model"""

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.lora_target_modules,
        bias=lora_args.bias,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def tokenize_prompt_response_pairs(prompt_texts, response_texts, tokenizer, max_length):
    """Pad to a fixed length and compute loss only on response tokens."""
    eos = tokenizer.eos_token or ""
    tokenized_full = tokenizer(
        [prompt + response + eos for prompt, response in zip(prompt_texts, response_texts)],
        truncation=True, max_length=max_length, padding="max_length", return_tensors=None,
    )
    tokenized_prompt = tokenizer(
        prompt_texts, truncation=True, max_length=max_length, padding="max_length", return_tensors=None,
    )
    labels = []
    for input_ids, attention_mask, prompt_attention_mask in zip(
        tokenized_full["input_ids"], tokenized_full["attention_mask"], tokenized_prompt["attention_mask"]
    ):
        label = list(input_ids)
        prompt_len = int(sum(prompt_attention_mask))
        for index in range(min(prompt_len, len(label))):
            label[index] = -100
        for index, mask in enumerate(attention_mask):
            if mask == 0:
                label[index] = -100
        labels.append(label)
    tokenized_full["labels"] = labels
    return tokenized_full


def preprocess_gsm8k(examples, tokenizer, max_length):
    questions = examples.get("question", [""] * len(examples["answer"]))
    return tokenize_prompt_response_pairs(
        [f"Q: {question}\nA: Let's think step by step. " for question in questions],
        [answer if answer is not None else "" for answer in examples["answer"]],
        tokenizer, max_length,
    )


def format_alpaca_prompt(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    content = f"{instruction}\n\n{input_text}" if input_text else instruction
    return f"### Instruction:\n{content}\n\n### Response:\n"


def preprocess_alpaca(examples, tokenizer, max_length):
    if "instruction" not in examples or "output" not in examples:
        raise ValueError("Alpaca preprocessing expects 'instruction' and 'output' columns.")
    instructions = examples["instruction"]
    inputs = examples.get("input", [""] * len(instructions))
    return tokenize_prompt_response_pairs(
        [format_alpaca_prompt(instruction, input_text) for instruction, input_text in zip(instructions, inputs)],
        [(output or "").strip() for output in examples["output"]],
        tokenizer, max_length,
    )


def _get_column(examples, names, default=None):
    for name in names:
        if name in examples:
            return examples[name]
    if default is not None:
        return default
    raise KeyError(f"Missing required dataset column. Tried: {names}")


def _looks_like_slot_name(text: str) -> bool:
    stripped = text.strip()
    return bool(re.fullmatch(r"[A-Za-z0-9_]+", stripped)) and ("_" in stripped or stripped.islower())


def _parse_massive_slots(utterance: str, annotated_utterance: Optional[object]):
    if not isinstance(annotated_utterance, str):
        return []
    slots = []
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


def preprocess_massive(examples, tokenizer, max_length, *, intent_label_names=None):
    utterances = _get_column(examples, ["utt", "utterance"])
    intents = _get_column(examples, ["intent_str", "intent_text", "intent"])
    annotated_utterances = _get_column(
        examples, ["annot_utt", "annotated_utt", "annotated_utterance"], [None] * len(utterances)
    )
    responses = []
    for utterance, intent, annotation in zip(utterances, intents, annotated_utterances):
        utterance = "" if utterance is None else str(utterance)
        if isinstance(intent, int) and intent_label_names and 0 <= intent < len(intent_label_names):
            intent = intent_label_names[intent]
        responses.append(json.dumps({"intent": "" if intent is None else str(intent),
                                     "slots": _parse_massive_slots(utterance, annotation)}, ensure_ascii=False, indent=2))
    return tokenize_prompt_response_pairs(
        [MASSIVE_PROMPT_TEMPLATE.format(utt="" if utterance is None else str(utterance)) for utterance in utterances],
        responses, tokenizer, max_length,
    )


def _load_massive_via_parquet_api(dataset_name: str, config_name: str, split: str):
    endpoints = [os.environ.get("HF_ENDPOINT", "").rstrip("/"), "https://huggingface.co"]
    last_error = None
    for endpoint in dict.fromkeys(endpoint for endpoint in endpoints if endpoint):
        url = f"{endpoint}/api/datasets/{dataset_name}/parquet/{config_name}/{split}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                files = json.loads(response.read().decode("utf-8"))
            if files:
                return load_dataset("parquet", data_files={split: files}, split=split)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
    raise RuntimeError(f"Failed to load MASSIVE fallback for {dataset_name}/{config_name} [{split}]") from last_error


def load_and_preprocess_dataset(data_args, tokenizer, max_length, dataset_kind: str, verbose: bool = True):
    if verbose:
        label = data_args.dataset_name if not data_args.dataset_config_name else f"{data_args.dataset_name}/{data_args.dataset_config_name}"
        print(f"Loading dataset: {label} [{data_args.dataset_split}]")
    try:
        if data_args.dataset_config_name is None:
            dataset = load_dataset(data_args.dataset_name, split=data_args.dataset_split)
        else:
            dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.dataset_split)
    except RuntimeError as error:
        if dataset_kind != "massive" or "Dataset scripts are no longer supported" not in str(error):
            raise
        dataset = _load_massive_via_parquet_api(data_args.dataset_name, data_args.dataset_config_name, data_args.dataset_split)
    if data_args.max_train_samples is not None:
        dataset = dataset.select(range(min(len(dataset), data_args.max_train_samples)))
    if dataset_kind == "gsm8k":
        preprocess = lambda batch: preprocess_gsm8k(batch, tokenizer, max_length)
    elif dataset_kind == "alpaca":
        preprocess = lambda batch: preprocess_alpaca(batch, tokenizer, max_length)
    else:
        intent_feature = getattr(dataset, "features", {}).get("intent")
        intent_names = list(intent_feature.names) if hasattr(intent_feature, "names") else None
        preprocess = lambda batch: preprocess_massive(batch, tokenizer, max_length, intent_label_names=intent_names)
    if verbose:
        print(f"Dataset size: {len(dataset)}")
    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names, desc=f"Tokenizing {dataset_kind}")


def main():
    """Main QLoRA training entry point for all three experiment datasets."""
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning on GSM8K, Alpaca, or MASSIVE")
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), default="gsm8k", help="Dataset profile and preprocessing format")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Base model path or HF identifier")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to the dataset profile)")
    parser.add_argument("--dataset_name", type=str, default=None, help="Optional Hugging Face dataset override")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Optional Hugging Face config override")
    parser.add_argument("--dataset_split", type=str, default=None, help="Optional dataset split override")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs (math reasoning needs more)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate (lower for math)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max_seq_length", type=int, default=768, help="Max sequence length (longer for math)")
    # LoRA arguments (with recommended defaults)
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (higher for complex reasoning)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    # DDP argument
    parser.add_argument("--local_rank", type=int, default=-1, help="For DDP training")

    args = parser.parse_args()

    # Get local rank for DDP
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    is_main_process = local_rank == 0

    # Initialize configs
    model_args = ModelArguments()
    data_args = DataArguments()
    lora_args = LoraArguments()
    train_args = TrainArgs()
    quant_args = QuantizationArguments()

    defaults = DATASET_DEFAULTS[args.dataset]
    data_args.dataset_name = args.dataset_name or defaults["dataset_name"]
    data_args.dataset_config_name = (
        args.dataset_config_name if args.dataset_config_name is not None else defaults["dataset_config_name"]
    )
    data_args.dataset_split = args.dataset_split or defaults["dataset_split"]
    if args.model_name_or_path:
        model_args.model_name_or_path = args.model_name_or_path

    # Override with command line arguments
    train_args.num_train_epochs = args.num_train_epochs
    train_args.learning_rate = args.learning_rate
    train_args.weight_decay = args.weight_decay
    train_args.warmup_ratio = args.warmup_ratio
    train_args.lr_scheduler_type = args.lr_scheduler_type
    train_args.output_dir = args.output_dir or defaults["output_dir"]
    train_args.per_device_train_batch_size = args.per_device_train_batch_size
    train_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    train_args.max_seq_length = args.max_seq_length

    lora_args.lora_r = args.lora_r
    lora_args.lora_alpha = args.lora_alpha
    lora_args.lora_dropout = args.lora_dropout

    # Only override these if explicitly provided
    if args.per_device_train_batch_size:
        train_args.per_device_train_batch_size = args.per_device_train_batch_size
    if args.gradient_accumulation_steps:
        train_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_train_samples:
        data_args.max_train_samples = args.max_train_samples
    if args.max_seq_length:
        train_args.max_seq_length = args.max_seq_length

    # Print config only on main process
    if is_main_process:
        print("=" * 80)
        print(f"QLoRA Fine-tuning on {args.dataset.upper()} Dataset")
        print("=" * 80)
        print(f"Model: {model_args.model_name_or_path}")
        dataset_label = data_args.dataset_name if not data_args.dataset_config_name else f"{data_args.dataset_name}/{data_args.dataset_config_name}"
        print(f"Dataset: {dataset_label} [{data_args.dataset_split}]")
        print(f"Output directory: {train_args.output_dir}")
        print("-" * 80)
        print("Training Configuration:")
        print(f"  Epochs: {train_args.num_train_epochs}")
        print(f"  Batch size per GPU: {train_args.per_device_train_batch_size}")
        print(f"  Gradient accumulation steps: {train_args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps * max(1, torch.cuda.device_count())}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  Learning rate: {train_args.learning_rate}")
        print(f"  Weight decay: {train_args.weight_decay}")
        print(f"  Warmup ratio: {train_args.warmup_ratio}")
        print(f"  LR scheduler: {train_args.lr_scheduler_type}")
        print(f"  Optimizer: {train_args.optim}")
        print(f"  Gradient checkpointing: {train_args.gradient_checkpointing}")
        print(f"  Max sequence length: {train_args.max_seq_length}")
        print(f"  Dataloader workers: {train_args.dataloader_num_workers}")
        print("-" * 80)
        print("LoRA Configuration:")
        print(f"  LoRA rank (r): {lora_args.lora_r}")
        print(f"  LoRA alpha: {lora_args.lora_alpha}")
        print(f"  LoRA dropout: {lora_args.lora_dropout}")
        print(f"  Target modules: {lora_args.lora_target_modules}")
        print("-" * 80)
        print("Quantization Configuration:")
        print(f"  Load in 4-bit: {quant_args.load_in_4bit}")
        print(f"  4-bit quant type: {quant_args.bnb_4bit_quant_type}")
        print(f"  4-bit compute dtype: {quant_args.bnb_4bit_compute_dtype}")
        print(f"  4-bit use double quant: {quant_args.bnb_4bit_use_double_quant}")
        print(f"  Flash Attention: {model_args.use_flash_attention}")
        print("=" * 80)

    # Pass local_rank to model loading function
    train_args.local_rank = local_rank

    # Load model and tokenizer
    if is_main_process:
        print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args, quant_args, train_args, local_rank)

    # Apply LoRA
    if is_main_process:
        print("\nApplying LoRA adapters...")
    model = apply_lora(model, lora_args)

    # Load and preprocess dataset
    if is_main_process:
        print(f"\nLoading and preprocessing {args.dataset.upper()} dataset...")
    train_dataset = load_and_preprocess_dataset(
        data_args,
        tokenizer,
        train_args.max_seq_length,
        args.dataset,
        verbose=is_main_process,
    )

    # Initialize Trainer
    warmup_kwargs = {}
    if train_args.warmup_ratio is not None:
        warmup_kwargs["warmup_ratio"] = train_args.warmup_ratio
    else:
        warmup_kwargs["warmup_steps"] = train_args.warmup_steps

    training_args = transformers.TrainingArguments(
        output_dir=train_args.output_dir,
        num_train_epochs=train_args.num_train_epochs,
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        learning_rate=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
        **warmup_kwargs,
        logging_steps=train_args.logging_steps,
        save_steps=train_args.save_steps,
        fp16=train_args.fp16,
        bf16=train_args.bf16,
        max_grad_norm=train_args.max_grad_norm,
        gradient_checkpointing=train_args.gradient_checkpointing,
        optim=train_args.optim,
        lr_scheduler_type=train_args.lr_scheduler_type,
        save_strategy="epoch",
        eval_strategy="no",
        load_best_model_at_end=False,
        #report_to=["tensorboard"],
        #save_safetensors=True,
        # DDP settings
        ddp_find_unused_parameters=train_args.ddp_find_unused_parameters,
        # Dataloader settings
        dataloader_num_workers=train_args.dataloader_num_workers,
        dataloader_prefetch_factor=train_args.dataloader_prefetch_factor,
        # Performance optimizations
        torch_compile=False,
    )

    epoch_timer_callback = EpochWallClockTimerCallback(train_args.output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[epoch_timer_callback],
    )

    # Train
    if is_main_process:
        print("\n" + "=" * 80)
        print(f"Starting training on {args.dataset.upper()}...")
        print("=" * 80)

    train_result = trainer.train()

    # Save final model
    if is_main_process:
        print("\nSaving final model...")
    trainer.save_model()
    trainer.save_state()

    # Save metrics
    metrics = train_result.metrics
    metrics["epoch_wall_clock_times"] = epoch_timer_callback.epoch_wall_clock_times
    epoch_times_path = None
    if trainer.is_world_process_zero():
        epoch_times_path = epoch_timer_callback.write()
        metrics["epoch_wall_clock_times_file"] = epoch_times_path
    if is_main_process:
        loggable_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        trainer.log_metrics("train", loggable_metrics)
    trainer.save_metrics("train", metrics)

    if is_main_process:
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Model saved to: {train_args.output_dir}")
        if epoch_times_path is not None:
            print(f"Epoch wall-clock times: {epoch_times_path}")
        print(f"Total training steps: {trainer.state.global_step if hasattr(trainer.state, 'global_step') else 'N/A'}")
        print(f"Final training loss: {metrics.get('train_loss', 'N/A')}")
        print("=" * 80)


if __name__ == "__main__":
    main()
