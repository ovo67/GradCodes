"""
Configuration for QLoRA fine-tuning of Llama-3.2-1B-Instruct
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Model related arguments"""
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-3B-Instruct",#"Qwen/Qwen3-0.6B",#"meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Use Flash Attention for faster training"}
    )


@dataclass
class DataArguments:
    """Data related arguments"""
    dataset_name: str = field(
        default="yahma/alpaca-cleaned",
        metadata={"help": "Dataset name from Hugging Face"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional dataset config name from Hugging Face"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of training examples"}
    )


@dataclass
class LoraArguments:
    """LoRA specific arguments"""
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension (rank)"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "List of module names to apply LoRA to"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA: 'none', 'all', 'lora_only'"}
    )


@dataclass
class TrainingArguments:
    """Training related arguments"""
    output_dir: str = field(
        default="./outputs/llama-3.2-1b-qlora-alpaca",
        metadata={"help": "Output directory for model checkpoints and logs"}
    )
    num_train_epochs: int = field(
        default=4,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing an update"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "Initial learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Linear warmup steps"}
    )
    warmup_ratio: float = field(
        default=None,
        metadata={"help": "Warmup ratio (fraction of total training steps). Overrides warmup_steps if set."}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use fp16 mixed precision training"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16 mixed precision training"}
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={"help": "Max gradient norm"}
    )
    max_seq_length: int = field(
        default=768,
        metadata={"help": "Maximum sequence length"}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to save memory"}
    )
    optim: str = field(
        default="adamw_torch_fused",
        metadata={"help": "Optimizer to use"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler"}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "When using DDP, find unused parameters"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of subprocesses for data loading"}
    )
    dataloader_prefetch_factor: int = field(
        default=2,
        metadata={"help": "Number of batches to prefetch in data loading"}
    )


@dataclass
class QuantizationArguments:
    """Quantization arguments for QLoRA"""
    load_in_4bit: bool = field(
        default=True,
        metadata={"help": "Load model in 4-bit quantization"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type: 'fp4' or 'nf4'"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Use nested quantization (double quantization)"}
    )
