# Grounded Answer Updating for Multi-Timestamp Video Question Answering

Grouped multi-timestamp video QA on RTV-Bench with three modes:
- **create splits** from RTV-Bench into `train/`, `dev/`, `test/`
- **evaluate** Qwen2.5-VL on grouped `q0/q1/q2` questions with or without JSON
- **train** a LoRA adapter with **GRPO** using grouped JSON outputs and a curriculum reward

## Project layout

```text
TemporalVQA/
├── Train/
│   └── train_rtv_bench_json_3qs_grouped_grpo_curriculum_reward.py
├── Test/
│   ├── rtv_bench_json_3qs_grouped_evaluation.py
│   └── rtv_bench_non_json_3qs_grouped_evaluation.py
├── create_train_dev_test_rtvbench.py
├── Dataset/
│   ├── train/
│   ├── dev/
│   └── test/
└── Results/
```

Each split directory must contain:

```text
<split>/
├── QA.json
└── videos/
```

## Install

System:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Python:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U "transformers>=4.49.0" "accelerate>=1.2.0" "datasets>=3.2.0" "peft>=0.14.0" "trl>=0.13.0"
pip install pillow numpy wandb sentencepiece huggingface_hub qwen-vl-utils
```

Optional:

```bash
huggingface-cli login
wandb login
```

## Quick script overview

### `create_train_dev_test_rtvbench.py`
Downloads/loads RTV-Bench, splits it into `train/dev/test`, writes `QA.json` per split, and places videos by symlink/copy/hardlink. Default split unit is **video**, which avoids leakage across splits.

### `Test/rtv_bench_non_json_3qs_grouped_evaluation.py`
Grouped evaluation without JSON. The model sees the video contact sheet plus grouped `q0/q1/q2` multiple-choice questions and must output one choice per question in free-form text. It writes grouped predictions, flat predictions, and a summary with **Accuracy** and strict group **Score**.

### `Test/rtv_bench_json_3qs_grouped_evaluation.py`
Grouped evaluation with short JSON output. The model sees the same grouped questions, but must return one JSON object with `final_answer` and `rationale` for each of `q0/q1/q2`. It also reports JSON-valid and schema-valid rates.

### `Train/train_rtv_bench_json_3qs_grouped_grpo_curriculum_reward.py`
GRPO training for the grouped JSON setting. It builds one training example per complete `q0/q1/q2` group, creates contact sheets from video prefixes, trains a LoRA adapter on Qwen2.5-VL, and saves the final adapter in `final_adapter/`.

## Metrics

- **Accuracy** = correctly answered questions / total questions
- **Score** = groups where **all three** (`q0`, `q1`, `q2`) are correct / total groups

## 1) Create train/dev/test splits

```bash
python create_train_dev_test_rtvbench.py \
  --output_root Dataset \
  --split_unit video \
  --train_ratio 0.60 \
  --dev_ratio 0.20 \
  --test_ratio 0.20 \
  --seed 42 \
  --video_mode symlink
```

Useful options:
- `--qa_json <path>`: use a local `QA.json`
- `--dataset_repo RTVBench/RTV-Bench`
- `--video_mode symlink|copy|hardlink|none`

## 2) Run non-JSON grouped evaluation

```bash
python Test/rtv_bench_non_json_3qs_grouped_evaluation.py \
  --test_dir Dataset/test \
  --output_dir Results/qwen25vl_base_non_json_grouped_test \
  --cache_dir cache/non_json_test \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --num_frames 8 \
  --max_new_tokens 1500 \
  --image_mode contact_sheet
```

Outputs:
- `grouped_predictions.json`
- `predictions_flat.json`
- `summary.json`

## 3) Run JSON grouped evaluation

```bash
python Test/rtv_bench_json_3qs_grouped_evaluation.py \
  --test_dir Dataset/test \
  --output_dir Results/qwen25vl_base_json_grouped_test \
  --cache_dir cache/json_test \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --num_frames 8 \
  --max_new_tokens 1500 \
  --image_mode contact_sheet
```

Outputs:
- `grouped_predictions.json`
- `predictions_flat.json`
- `summary.json`

## 4) Train with GRPO

```bash
accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  Train/train_rtv_bench_json_3qs_grouped_grpo_curriculum_reward.py \
  --train_dir Dataset/train \
  --output_dir runs/qwen25vl7b_grpo_json_grouped \
  --cache_dir cache/grpo_train \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --wandb_project TemporalVQA-GRPO-Qwen25VL \
  --wandb_run_name qwen25vl7b-grpo-json-grouped \
  --wandb_mode online \
  --num_frames 8 \
  --num_generations 4 \
  --generation_batch_size 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_completion_length 1024 \
  --num_train_epochs 5 \
  --save_steps 100 \
  --logging_steps 5 \
  --seed 10 \
  --bf16
```

Training output:
- checkpoints under `--output_dir`
- `run_config.json`
- `final_adapter/`

## 5) Evaluate the trained adapter

```bash
python Test/rtv_bench_json_3qs_grouped_evaluation.py \
  --test_dir Dataset/test \
  --output_dir Results/qwen25vl_grpo_json_grouped_test \
  --cache_dir cache/json_eval_trained \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter_path runs/qwen25vl7b_grpo_json_grouped/final_adapter \
  --device cuda \
  --dtype bfloat16 \
  --num_frames 8 \
  --max_new_tokens 1500 \
  --image_mode contact_sheet
```

## Notes

- Use `contact_sheet` mode to match training.
- Both eval scripts support `--resume`.
- `ffmpeg` and `ffprobe` must be available on `PATH`.
