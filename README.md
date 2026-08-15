# Gradcodes

Gradcodes is a self-contained research codebase for discrete search
on quantized language models. It includes the main Gradcodes method, two
dataset-specific evaluation pipelines, and a unified QLoRA baseline.

Install the runtime dependencies with:

```bash
pip install -r requirements.txt
```

## Repository layout

- `src/`: Gradcodes core implementation and the unified training entry point.
- `eval/`: evaluation programs for GSM8K, AlpacaEval, and MASSIVE.
- `baselines/`: unified QLoRA baseline and its local configuration module.
- `docs/`: supplementary evaluation notes.

## Gradcodes training

`src/train_gradcodes.py` trains on GSM8K, Alpaca, or MASSIVE using a common
search, quantization, checkpointing, and resume workflow. The `--dataset`
profile selects the preprocessing function and default dataset source.

```bash
# Replace gsm8k with alpaca or massive for the other training profiles.
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  src/train_gradcodes.py --dataset gsm8k
```

Use `--dataset_name`, `--dataset_config_name`, `--dataset_split`, and
`--output_dir` to override profile defaults. Checkpoints are saved as
`gradcodes_state.pt`; resume a run with
`--resume_from_checkpoint <checkpoint-directory-or-state-file>`.

## Evaluation

```bash
# GSM8K with lm-eval
python eval/eval_lm.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --gradcodes_artifact outputs/llama-3.2-1b-gradcodes-gsm8k-nf4/epoch_0002 \
  --tasks gsm8k_cot_zeroshot --batch_size 8

# MASSIVE en-US test set
python eval/eval_massive.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --gradcodes_artifact outputs/llama-3.2-1b-gradcodes-massive/epoch_0005
```

## baselines

`baselines/train_qlora.py` uses the same dataset profiles and preprocessing
formats as the main training entry point.

```bash
# Replace gsm8k with alpaca or massive for the other profiles.
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  baselines/train_qlora.py --dataset gsm8k
```

Use `--model_name_or_path`, `--dataset_name`, `--dataset_config_name`,
`--dataset_split`, and `--output_dir` to override the defaults.
