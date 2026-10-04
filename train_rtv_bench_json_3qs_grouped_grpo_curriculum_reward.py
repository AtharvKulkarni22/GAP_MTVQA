#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from PIL import Image, ImageDraw, ImageOps
from peft import LoraConfig
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

try:
    import wandb
except Exception:
    wandb = None

try:
    from trl import GRPOTrainer, GRPOConfig
except Exception:
    from trl.trainer.grpo_trainer import GRPOTrainer
    from trl.trainer.grpo_config import GRPOConfig


GROUP_ID_RE = re.compile(r"^q-group-([a-z0-9]+)-([012])(?:-[^-]+)*$", re.IGNORECASE)
FINAL_ANSWER_JSON_RE = re.compile(r'"final_answer"\s*:\s*"([A-D])"', re.IGNORECASE)
LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)

MAIN_CATEGORIES = ("Object", "Action", "Event")
DIM_CATEGORIES = ("Perception", "Understanding", "Reasoning")
VALID_LEVELS = {"q0", "q1", "q2"}
VALID_QUESTION_TYPES = {
    "attribute", "count", "action", "event", "intent", "future", "reasoning", "other"
}
VALID_ROLES = {"primary", "secondary"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_SUPPORT_SUFFICIENCY = {"sufficient", "partial", "insufficient"}
VALID_DEPENDENCY_TYPES = {
    "direct_visual_evidence", "counting", "temporal_change", "causal_inference", "scoreboard", "other"
}
VALID_UPDATE_STATUS = {"unknown", "stay", "update"}

PLACEHOLDER_PATTERNS = [
    "short string",
    "short fact",
    "brief grounded explanation",
    "short grounded state",
    "true/false",
    "stay|update|unknown",
    "entity 1",
    "entity 2",
]

CURRENT_EPOCH = 0.0
CURRENT_GLOBAL_STEP = 0
TOTAL_TRAIN_STEPS = 1
WANDB_RUN = None
WANDB_STEP_CACHE: Dict[str, List[float]] = defaultdict(list)
WANDB_PROJECT_DEFAULT = "TemporalVQA-GRPO-Qwen25VL"
SCRIPT_PATH = os.path.abspath(__file__)
_DURATION_CACHE: Dict[str, float] = {}


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0


def get_local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        return 0


def is_main_process() -> bool:
    return get_rank() == 0


def log(msg: str) -> None:
    prefix = (
        f"[{now_str()}]"
        f"[host={socket.gethostname()}]"
        f"[pid={os.getpid()}]"
        f"[rank={get_rank()}]"
        f"[local_rank={get_local_rank()}]"
    )
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def log_exception(context: str) -> None:
    log(f"EXCEPTION in {context}")
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "unknown")
    root = f"/scratch/general/vast/{user}/TemporalVQA"

    p = argparse.ArgumentParser("TemporalVQA GRPO training with grouped q0/q1/q2 short-schema reward curriculum")
    p.add_argument("--train_dir", type=str, default=f"{root}/Dataset/train")
    p.add_argument("--output_dir", type=str, default=f"{root}/grpo_runs/qwen25vl7b_grpo_grouped_short_schema_curriculum_reward_5ep")
    p.add_argument("--cache_dir", type=str, default=f"{root}/grpo_cache")
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")

    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--max_examples", type=int, default=None)

    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--frame_width", type=int, default=448)
    p.add_argument("--frame_height", type=int, default=448)

    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--num_train_epochs", type=float, default=5.0)

    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_total_limit", type=int, default=20)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--generation_batch_size", type=int, default=4)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=512)

    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)

    # Reward curriculum over one grouped q0/q1/q2 rollout.
    # JSON + short-schema validity are hard gates; if either fails, reward is 0.
    # 0%-35%: 0.15*q0 + 0.25*q1 + 0.40*q2 + 0.20*all_correct
    # 35%-70%: 0.10*q0 + 0.20*q1 + 0.35*q2 + 0.35*all_correct
    # 70%-100%: 1.00*all_correct only
    p.add_argument("--reward_weight_consistency", type=float, default=None)
    p.add_argument("--reward_weight_answer", type=float, default=None)
    p.add_argument("--reward_weight_json_alignment", type=float, default=None)

    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--scale_rewards", type=str, default="batch")

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="bf16")
    p.add_argument("--fp16", action="store_true", default=False)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    p.add_argument("--wandb_project", type=str, default=WANDB_PROJECT_DEFAULT)
    p.add_argument("--wandb_run_name", type=str, default="qwen25vl-grpo-grouped-short-schema-curriculum-5ep")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def parse_group_id_and_qtype(question_id: Any) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(question_id, str):
        return None, None
    m = GROUP_ID_RE.match(question_id.strip())
    if not m:
        return None, None
    return m.group(1), m.group(2)


def parse_main_category(type_str: Any) -> Optional[str]:
    if not isinstance(type_str, str):
        return None
    for cat in MAIN_CATEGORIES:
        if type_str.startswith(f"{cat}-"):
            return cat
    return None


def parse_dimension(type_str: Any) -> Optional[str]:
    if not isinstance(type_str, str):
        return None
    if ("TP" in type_str) or ("VP" in type_str) or ("SP" in type_str):
        return "Perception"
    if ("IA" in type_str) or ("PU" in type_str) or ("GU" in type_str):
        return "Understanding"
    if ("SR" in type_str) or ("FP" in type_str):
        return "Reasoning"
    return None


def infer_question_type(question: str, type_str: str) -> str:
    q = (question or "").lower()
    if "how many" in q or "number of" in q or "count" in q:
        return "count"
    if "what color" in q or "which color" in q or "color of" in q:
        return "attribute"
    if "what will" in q or "happen next" in q or "about to" in q:
        return "future"
    if "why" in q or "purpose" in q or "who might" in q:
        return "intent"
    if "doing" in q or q.startswith("is ") or q.startswith("are "):
        return "action"
    if parse_dimension(type_str) == "Reasoning":
        return "reasoning"
    if parse_main_category(type_str) == "Event":
        return "event"
    return "other"


def sort_key(sample: Dict[str, Any]) -> Tuple[float, str, str]:
    _, qtype = parse_group_id_and_qtype(sample.get("questionID", ""))
    return (float(sample.get("end_time", 0.0)), str(qtype), sample.get("questionID", ""))


def build_previous_timestamp_map(items: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        gid, _ = parse_group_id_and_qtype(item.get("questionID", ""))
        key = gid if gid is not None else f"video::{item.get('video', '')}"
        grouped[key].append(item)

    prev_map: Dict[str, Optional[float]] = {}
    for _, group_items in grouped.items():
        group_items = sorted(group_items, key=sort_key)
        prev_t: Optional[float] = None
        for item in group_items:
            prev_map[item["questionID"]] = prev_t
            prev_t = float(item.get("end_time", 0.0))
    return prev_map


def build_option_prompt(options: Dict[str, str]) -> str:
    keys = sorted(options.keys())
    return "Choices:\n" + "\n".join([f"{k}. {options[k]}" for k in keys])

def _single_question_block_for_group(
    item: Dict[str, Any],
    qlevel: str,
    question_type: str,
    previous_timestamp: Optional[float],
) -> str:
    prev_ts = "null" if previous_timestamp is None else f"{float(previous_timestamp):.3f}"
    options = item["options"]
    return f"""
{qlevel.upper()}:
question_id: {item["questionID"]}
queried_timestamp: {float(item["end_time"]):.3f}
previous_queried_timestamp: {prev_ts}
expected_question_type: {question_type}
question: {item["question"]}
{build_option_prompt(options)}
""".strip()


def build_grouped_temporal_json_prompt(
    q0_item: Dict[str, Any],
    q1_item: Dict[str, Any],
    q2_item: Dict[str, Any],
    q0_question_type: str,
    q1_question_type: str,
    q2_question_type: str,
    q0_previous_timestamp: Optional[float],
    q1_previous_timestamp: Optional[float],
    q2_previous_timestamp: Optional[float],
    latest_end_time: float,
) -> str:
    q0_block = _single_question_block_for_group(q0_item, "q0", q0_question_type, q0_previous_timestamp)
    q1_block = _single_question_block_for_group(q1_item, "q1", q1_question_type, q1_previous_timestamp)
    q2_block = _single_question_block_for_group(q2_item, "q2", q2_question_type, q2_previous_timestamp)

    return f"""
You are a careful temporal video reasoning assistant.

You are given video frames sampled in chronological order from time 0 up to {latest_end_time:.3f} seconds.
The frame labels show approximate timestamps. Answer each question using the scene state at that question's own queried_timestamp.

You must answer all three related hierarchy questions q0, q1, and q2 for the same temporal group.

Questions:
{q0_block}

{q1_block}

{q2_block}

Return exactly one valid JSON object and nothing else.
Use this short schema exactly:
{{
  "q0": {{
    "final_answer": "A",
    "rationale": "short visual evidence"
  }},
  "q1": {{
    "final_answer": "B",
    "rationale": "short visual evidence"
  }},
  "q2": {{
    "final_answer": "C",
    "rationale": "short temporal reasoning"
  }}
}}

Rules:
1. The top-level JSON object must have keys "q0", "q1", and "q2".
2. For each question, final_answer must be exactly one of that question's choices: A, B, C, or D.
3. Each rationale must be a short grounded sentence, preferably under 20 words.
4. Do not output markdown, code fences, extra text, or multiple JSON objects.
5. If uncertain, still choose the best option from the given choices.
""".strip()

def get_video_duration(video_path: str) -> float:
    if video_path in _DURATION_CACHE:
        return _DURATION_CACHE[video_path]
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {proc.stderr.strip()}")
    duration = float(proc.stdout.strip())
    _DURATION_CACHE[video_path] = duration
    return duration


def sample_timestamps(end_time: float, num_frames: int, video_duration: Optional[float] = None) -> List[float]:
    end_time = max(float(end_time), 0.05)
    if video_duration is not None:
        end_time = min(end_time, max(0.05, video_duration - 0.25))
    if num_frames <= 1:
        return [end_time]
    return [end_time * (i + 1) / num_frames for i in range(num_frames)]


def wait_for_file(path: Path, timeout: float = 120.0, min_size: int = 1024) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if path.exists() and path.stat().st_size >= min_size:
                return True
        except FileNotFoundError:
            pass
        time.sleep(0.25)
    return False


def acquire_lock(lock_dir: Path, wait_timeout: float = 600.0) -> bool:
    start = time.time()
    while time.time() - start < wait_timeout:
        try:
            os.mkdir(lock_dir)
            return True
        except FileExistsError:
            time.sleep(0.5)
    return False


def release_lock(lock_dir: Path) -> None:
    try:
        shutil.rmtree(lock_dir)
    except Exception:
        pass


def extract_frame_atomic(video_path: str, ts: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration(video_path)
    safe_ts = min(max(0.0, ts), max(0.0, duration - 0.25))
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp.{os.getpid()}{out_path.suffix}")

    def _run_ffmpeg(seek_ts: float) -> None:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{seek_ts:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            str(tmp_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {video_path} at ts={seek_ts:.3f}: {proc.stderr.strip()}")

    tried: List[float] = []
    for candidate_ts in [safe_ts, max(0.0, safe_ts - 1.0), max(0.0, safe_ts - 2.0)]:
        if any(abs(candidate_ts - old) < 1e-6 for old in tried):
            continue
        tried.append(candidate_ts)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        try:
            _run_ffmpeg(candidate_ts)
        except Exception:
            continue
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            os.replace(tmp_path, out_path)
            return

    raise RuntimeError(
        f"ffmpeg did not create frame for {video_path} at ts={ts:.3f} "
        f"(duration={duration:.3f}, tried={tried})"
    )


def make_contact_sheet(
    video_path: str,
    end_time: float,
    num_frames: int,
    width: int,
    height: int,
    out_path: Path,
) -> Path:
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = out_path.parent / f"{out_path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = out_path.parent / f"{out_path.stem}.lock"

    if not acquire_lock(lock_dir):
        raise RuntimeError(f"Could not acquire lock for {out_path}")

    try:
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        duration = get_video_duration(video_path)
        timestamps = sample_timestamps(end_time, num_frames, video_duration=duration)
        frame_paths: List[Path] = []

        for i, ts in enumerate(timestamps):
            fp = frame_dir / f"frame_{i:02d}.jpg"
            if not (fp.exists() and fp.stat().st_size > 0):
                extract_frame_atomic(video_path, ts, fp)
            if not wait_for_file(fp, timeout=30.0, min_size=1024):
                raise RuntimeError(f"Timed out waiting for frame: {fp}")
            frame_paths.append(fp)

        cols = min(4, num_frames)
        rows = math.ceil(num_frames / cols)
        sheet = Image.new("RGB", (cols * width, rows * height), color=(255, 255, 255))
        draw = ImageDraw.Draw(sheet)

        for idx, (ts, fp) in enumerate(zip(timestamps, frame_paths)):
            with Image.open(fp) as img:
                img = img.convert("RGB")
                img = ImageOps.fit(img, (width, height))
                x = (idx % cols) * width
                y = (idx // cols) * height
                sheet.paste(img, (x, y))
                draw.rectangle([(x, y), (x + 110, y + 28)], fill=(255, 255, 255))
                draw.text((x + 6, y + 6), f"t={ts:.1f}s", fill=(0, 0, 0))

        tmp_sheet = out_path.with_name(f"{out_path.stem}.tmp.{os.getpid()}{out_path.suffix}")
        if tmp_sheet.exists():
            try:
                tmp_sheet.unlink()
            except Exception:
                pass
        sheet.save(tmp_sheet, quality=95)
        if not tmp_sheet.exists() or tmp_sheet.stat().st_size == 0:
            raise RuntimeError(f"Failed to create contact sheet temp file: {tmp_sheet}")
        os.replace(tmp_sheet, out_path)
        return out_path
    finally:
        release_lock(lock_dir)


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


def load_train_records(train_dir: Path, cache_dir: Path, num_frames: int, frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
    """Load one training record per complete q0/q1/q2 group."""
    qa_path = train_dir / "QA.json"
    videos_dir = train_dir / "videos"
    if not qa_path.exists():
        raise FileNotFoundError(f"Missing QA.json: {qa_path}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing videos dir: {videos_dir}")

    with open(qa_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{qa_path} must contain a list")

    prev_map = build_previous_timestamp_map(items)
    grouped_items: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    skipped_bad_group_id = 0

    for item in items:
        gid, qtype_num = parse_group_id_and_qtype(item.get("questionID", ""))
        if gid is None or qtype_num not in {"0", "1", "2"}:
            skipped_bad_group_id += 1
            continue
        grouped_items[gid][f"q{qtype_num}"] = item

    contact_root = cache_dir / "grouped_short_schema_contact_sheets"
    records: List[Dict[str, Any]] = []
    skipped_incomplete = 0

    for gid, qdict in sorted(grouped_items.items()):
        if not {"q0", "q1", "q2"}.issubset(qdict.keys()):
            skipped_incomplete += 1
            continue

        q0_item = qdict["q0"]
        q1_item = qdict["q1"]
        q2_item = qdict["q2"]

        video_name = q2_item["video"]
        video_path = (videos_dir / video_name).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Broken or missing video symlink: {videos_dir / video_name}")

        latest_end_time = max(
            float(q0_item.get("end_time", 0.0)),
            float(q1_item.get("end_time", 0.0)),
            float(q2_item.get("end_time", 0.0)),
        )
        q0_question_type = infer_question_type(q0_item["question"], str(q0_item.get("type", "")))
        q1_question_type = infer_question_type(q1_item["question"], str(q1_item.get("type", "")))
        q2_question_type = infer_question_type(q2_item["question"], str(q2_item.get("type", "")))

        prompt_text = build_grouped_temporal_json_prompt(
            q0_item=q0_item,
            q1_item=q1_item,
            q2_item=q2_item,
            q0_question_type=q0_question_type,
            q1_question_type=q1_question_type,
            q2_question_type=q2_question_type,
            q0_previous_timestamp=prev_map.get(q0_item["questionID"]),
            q1_previous_timestamp=prev_map.get(q1_item["questionID"]),
            q2_previous_timestamp=prev_map.get(q2_item["questionID"]),
            latest_end_time=latest_end_time,
        )

        image_name = f"group_{safe_slug(gid)}_{latest_end_time:.3f}.jpg"
        image_path = contact_root / image_name
        make_contact_sheet(
            video_path=str(video_path),
            end_time=latest_end_time,
            num_frames=num_frames,
            width=frame_width,
            height=frame_height,
            out_path=image_path,
        )
        if not wait_for_file(image_path, timeout=60.0, min_size=1024):
            raise RuntimeError(f"Timed out waiting for contact sheet: {image_path}")

        records.append(
            {
                "group_id": gid,
                "prompt_text": prompt_text,
                "image_path": str(image_path),
                "video": video_name,
                "latest_end_time": latest_end_time,
                "q0_questionID": q0_item["questionID"],
                "q1_questionID": q1_item["questionID"],
                "q2_questionID": q2_item["questionID"],
                "q0_answer": str(q0_item["answer"]).strip().upper(),
                "q1_answer": str(q1_item["answer"]).strip().upper(),
                "q2_answer": str(q2_item["answer"]).strip().upper(),
                "q0_options": q0_item["options"],
                "q1_options": q1_item["options"],
                "q2_options": q2_item["options"],
                "q0_end_time": float(q0_item.get("end_time", 0.0)),
                "q1_end_time": float(q1_item.get("end_time", 0.0)),
                "q2_end_time": float(q2_item.get("end_time", 0.0)),
                "q0_type": q0_item.get("type", ""),
                "q1_type": q1_item.get("type", ""),
                "q2_type": q2_item.get("type", ""),
            }
        )

    if is_main_process():
        log(
            f"Loaded {len(records)} complete q0/q1/q2 groups "
            f"(skipped_incomplete={skipped_incomplete}, skipped_bad_group_id={skipped_bad_group_id})"
        )
    return records


def example_to_vlm_record(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    images = []
    prompts = []
    for image_path, prompt_text in zip(batch["image_path"], batch["prompt_text"]):
        with Image.open(image_path) as img:
            image = img.convert("RGB").copy()
        images.append(image)
        prompts.append([
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ])

    return {
        "prompt": prompts,
        "image": images,
        "group_id": batch["group_id"],
        "video": batch["video"],
        "latest_end_time": batch["latest_end_time"],
        "q0_questionID": batch["q0_questionID"],
        "q1_questionID": batch["q1_questionID"],
        "q2_questionID": batch["q2_questionID"],
        "q0_answer": batch["q0_answer"],
        "q1_answer": batch["q1_answer"],
        "q2_answer": batch["q2_answer"],
        "q0_options": batch["q0_options"],
        "q1_options": batch["q1_options"],
        "q2_options": batch["q2_options"],
        "q0_end_time": batch["q0_end_time"],
        "q1_end_time": batch["q1_end_time"],
        "q2_end_time": batch["q2_end_time"],
        "q0_type": batch["q0_type"],
        "q1_type": batch["q1_type"],
        "q2_type": batch["q2_type"],
    }


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for x in content:
                if isinstance(x, dict) and isinstance(x.get("text"), str):
                    parts.append(x["text"])
                else:
                    parts.append(str(x))
            return "\n".join(parts)
        return str(completion)
    if isinstance(completion, list):
        parts: List[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for x in content:
                        if isinstance(x, dict) and isinstance(x.get("text"), str):
                            parts.append(x["text"])
                        else:
                            parts.append(str(x))
                else:
                    parts.append(str(item))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def extract_json_substring(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    return text[start:end + 1]


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    payload = extract_json_substring(text)
    if payload is None:
        return None
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _mean_or_zero(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def validate_grouped_schema(obj: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, bool], List[str]]:
    """Validate the exact short grouped JSON schema.

    Required structure:
      {
        "q0": {"final_answer": "A|B|C|D", "rationale": "non-empty string"},
        "q1": {"final_answer": "A|B|C|D", "rationale": "non-empty string"},
        "q2": {"final_answer": "A|B|C|D", "rationale": "non-empty string"}
      }

    Extra top-level keys or extra per-question keys are rejected so the reward
    matches the prompt's "use this short schema exactly" instruction.
    """
    errors: List[str] = []
    per_q_schema_valid: Dict[str, bool] = {"q0": False, "q1": False, "q2": False}
    required_top = {"q0", "q1", "q2"}
    required_sub = {"final_answer", "rationale"}

    if not isinstance(obj, dict):
        return False, per_q_schema_valid, ["not_a_dict"]

    top_keys = set(obj.keys())
    missing_top = required_top - top_keys
    extra_top = top_keys - required_top
    for key in sorted(missing_top):
        errors.append(f"missing_top:{key}")
    for key in sorted(extra_top):
        errors.append(f"extra_top:{key}")

    for qk in ("q0", "q1", "q2"):
        sub = obj.get(qk)
        q_errors: List[str] = []
        if not isinstance(sub, dict):
            q_errors.append("not_a_dict")
        else:
            sub_keys = set(sub.keys())
            missing_sub = required_sub - sub_keys
            extra_sub = sub_keys - required_sub
            for key in sorted(missing_sub):
                q_errors.append(f"missing:{key}")
            for key in sorted(extra_sub):
                q_errors.append(f"extra:{key}")

            fa = sub.get("final_answer")
            if not isinstance(fa, str) or fa.strip().upper() not in {"A", "B", "C", "D"}:
                q_errors.append("bad:final_answer")

            rationale = sub.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                q_errors.append("bad:rationale")

        per_q_schema_valid[qk] = len(q_errors) == 0
        errors.extend([f"{qk}:{e}" for e in q_errors])

    return len(errors) == 0 and all(per_q_schema_valid.values()), per_q_schema_valid, errors


def extract_grouped_pred_letters(text: str, q0_options: Dict[str, str], q1_options: Dict[str, str], q2_options: Dict[str, str]) -> Dict[str, str]:
    parsed = try_parse_json(text)
    preds = {"q0": "ERROR", "q1": "ERROR", "q2": "ERROR"}
    if isinstance(parsed, dict):
        for qk, opts in (("q0", q0_options), ("q1", q1_options), ("q2", q2_options)):
            sub = parsed.get(qk)
            if isinstance(sub, dict):
                fa = sub.get("final_answer")
                if isinstance(fa, str) and fa.strip().upper() in opts:
                    preds[qk] = fa.strip().upper()
    return preds


def _rationale_present(obj: Optional[Dict[str, Any]], qk: str) -> float:
    if not isinstance(obj, dict):
        return 0.0
    sub = obj.get(qk)
    if not isinstance(sub, dict):
        return 0.0
    rationale = sub.get("rationale")
    return 1.0 if isinstance(rationale, str) and bool(rationale.strip()) else 0.0


def get_reward_curriculum_weights() -> Tuple[float, float, float, float, float, str]:
    total = max(1, int(TOTAL_TRAIN_STEPS))
    progress = min(1.0, max(0.0, float(CURRENT_GLOBAL_STEP) / float(total)))
    if progress < 0.35:
        return 0.15, 0.25, 0.40, 0.20, progress, "phase_0_35_dense"
    if progress < 0.70:
        return 0.10, 0.20, 0.35, 0.35, progress, "phase_35_70_mixed"
    return 0.0, 0.0, 0.0, 1.0, progress, "phase_70_100_all_correct"


def strict_json_reward(
    completions,
    q0_answer,
    q1_answer,
    q2_answer,
    q0_options=None,
    q1_options=None,
    q2_options=None,
    group_id=None,
    **kwargs,
) -> List[float]:
    rewards: List[float] = []
    w_q0, w_q1, w_q2, w_all, progress, phase = get_reward_curriculum_weights()

    batch_json_valid = 0
    batch_group_schema_valid = 0
    batch_parsed_all_answers = 0
    gated_zero_json = 0
    gated_zero_schema = 0

    q0_corrects: List[float] = []
    q1_corrects: List[float] = []
    q2_corrects: List[float] = []
    all_corrects: List[float] = []
    q0_schema_valids: List[float] = []
    q1_schema_valids: List[float] = []
    q2_schema_valids: List[float] = []
    q0_rationale_present: List[float] = []
    q1_rationale_present: List[float] = []
    q2_rationale_present: List[float] = []

    for idx in range(len(completions)):
        text = completion_to_text(completions[idx])
        parsed = try_parse_json(text)
        json_valid = isinstance(parsed, dict)
        if json_valid:
            batch_json_valid += 1
        else:
            gated_zero_json += 1
            rewards.append(0.0)
            q0_corrects.append(0.0); q1_corrects.append(0.0); q2_corrects.append(0.0); all_corrects.append(0.0)
            q0_schema_valids.append(0.0); q1_schema_valids.append(0.0); q2_schema_valids.append(0.0)
            q0_rationale_present.append(0.0); q1_rationale_present.append(0.0); q2_rationale_present.append(0.0)
            continue

        group_schema_valid, per_q_schema_valid, _ = validate_grouped_schema(parsed)
        q0_schema_valids.append(float(per_q_schema_valid.get("q0", False)))
        q1_schema_valids.append(float(per_q_schema_valid.get("q1", False)))
        q2_schema_valids.append(float(per_q_schema_valid.get("q2", False)))
        q0_rationale_present.append(_rationale_present(parsed, "q0"))
        q1_rationale_present.append(_rationale_present(parsed, "q1"))
        q2_rationale_present.append(_rationale_present(parsed, "q2"))

        if not group_schema_valid:
            gated_zero_schema += 1
            rewards.append(0.0)
            q0_corrects.append(0.0); q1_corrects.append(0.0); q2_corrects.append(0.0); all_corrects.append(0.0)
            continue

        batch_group_schema_valid += 1
        opts0 = q0_options[idx] if q0_options is not None else {}
        opts1 = q1_options[idx] if q1_options is not None else {}
        opts2 = q2_options[idx] if q2_options is not None else {}
        preds = extract_grouped_pred_letters(text, opts0, opts1, opts2)

        if all(preds[qk] != "ERROR" for qk in ("q0", "q1", "q2")):
            batch_parsed_all_answers += 1

        q0c = 1.0 if preds["q0"] == str(q0_answer[idx]).strip().upper() else 0.0
        q1c = 1.0 if preds["q1"] == str(q1_answer[idx]).strip().upper() else 0.0
        q2c = 1.0 if preds["q2"] == str(q2_answer[idx]).strip().upper() else 0.0
        allc = 1.0 if (q0c > 0 and q1c > 0 and q2c > 0) else 0.0

        reward = (w_q0 * q0c) + (w_q1 * q1c) + (w_q2 * q2c) + (w_all * allc)
        rewards.append(float(reward))
        q0_corrects.append(q0c); q1_corrects.append(q1c); q2_corrects.append(q2c); all_corrects.append(allc)

    if is_main_process():
        denom = max(1, len(rewards))
        WANDB_STEP_CACHE["reward"].extend(rewards)
        WANDB_STEP_CACHE["json_valid_rate"].append(batch_json_valid / denom)
        WANDB_STEP_CACHE["group_schema_valid_rate"].append(batch_group_schema_valid / denom)
        WANDB_STEP_CACHE["schema_valid_q0"].append(_mean_or_zero(q0_schema_valids))
        WANDB_STEP_CACHE["schema_valid_q1"].append(_mean_or_zero(q1_schema_valids))
        WANDB_STEP_CACHE["schema_valid_q2"].append(_mean_or_zero(q2_schema_valids))
        WANDB_STEP_CACHE["rationale_present_q0"].append(_mean_or_zero(q0_rationale_present))
        WANDB_STEP_CACHE["rationale_present_q1"].append(_mean_or_zero(q1_rationale_present))
        WANDB_STEP_CACHE["rationale_present_q2"].append(_mean_or_zero(q2_rationale_present))
        WANDB_STEP_CACHE["parsed_all_answers_rate"].append(batch_parsed_all_answers / denom)
        WANDB_STEP_CACHE["answer_correct_q0"].append(_mean_or_zero(q0_corrects))
        WANDB_STEP_CACHE["answer_correct_q1"].append(_mean_or_zero(q1_corrects))
        WANDB_STEP_CACHE["answer_correct_q2"].append(_mean_or_zero(q2_corrects))
        WANDB_STEP_CACHE["all_q0_q1_q2_correct_rate"].append(_mean_or_zero(all_corrects))
        WANDB_STEP_CACHE["gate_zero_json_rate"].append(gated_zero_json / denom)
        WANDB_STEP_CACHE["gate_zero_schema_rate"].append(gated_zero_schema / denom)
        WANDB_STEP_CACHE["curriculum_progress"].append(float(progress))
        WANDB_STEP_CACHE["curriculum_weight_q0"].append(float(w_q0))
        WANDB_STEP_CACHE["curriculum_weight_q1"].append(float(w_q1))
        WANDB_STEP_CACHE["curriculum_weight_q2"].append(float(w_q2))
        WANDB_STEP_CACHE["curriculum_weight_all_correct"].append(float(w_all))
        phase_id = 0.0 if phase.startswith("phase_0") else (1.0 if phase.startswith("phase_35") else 2.0)
        WANDB_STEP_CACHE["curriculum_phase_id"].append(phase_id)

    return rewards


class EpochTrackerCallback(TrainerCallback):
    def _set_state(self, state) -> None:
        global CURRENT_EPOCH, CURRENT_GLOBAL_STEP, TOTAL_TRAIN_STEPS
        if state.epoch is not None:
            try:
                CURRENT_EPOCH = float(state.epoch)
            except Exception:
                pass
        try:
            CURRENT_GLOBAL_STEP = int(state.global_step)
        except Exception:
            pass
        try:
            if getattr(state, "max_steps", None) is not None and int(state.max_steps) > 0:
                TOTAL_TRAIN_STEPS = int(state.max_steps)
        except Exception:
            pass

    def on_train_begin(self, args, state, control, **kwargs):
        self._set_state(state)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._set_state(state)

    def on_step_begin(self, args, state, control, **kwargs):
        self._set_state(state)

    def on_step_end(self, args, state, control, **kwargs):
        self._set_state(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._set_state(state)


class WandbMetricsCallback(TrainerCallback):
    """Logs only custom reward metrics. Trainer/TRL owns normal W&B metrics."""

    def _flush_custom_metrics(self, state, force: bool = False) -> None:
        if not is_main_process() or WANDB_RUN is None or wandb is None or not WANDB_STEP_CACHE:
            return
        if not any(bool(vals) for vals in WANDB_STEP_CACHE.values()):
            return

        payload: Dict[str, Any] = {
            "custom/epoch": float(state.epoch) if state.epoch is not None else float(CURRENT_EPOCH),
            "custom/trainer_global_step": int(state.global_step),
            "custom/total_train_steps": int(TOTAL_TRAIN_STEPS),
            "custom/curriculum_progress_from_callback": float(
                min(1.0, max(0.0, int(state.global_step) / max(1, int(TOTAL_TRAIN_STEPS))))
            ),
        }
        for key, vals in list(WANDB_STEP_CACHE.items()):
            if not vals:
                continue
            payload[f"custom/{key}_mean"] = _mean_or_zero(vals)
            if key == "reward":
                arr = np.array(vals, dtype=np.float32)
                payload["custom/reward_std"] = float(arr.std())
                payload["custom/reward_max"] = float(arr.max())
                payload["custom/reward_min"] = float(arr.min())
                payload["custom/reward_p95"] = float(np.percentile(arr, 95))
                try:
                    payload["custom/reward_hist"] = wandb.Histogram(arr)
                except Exception:
                    pass

        wandb.log(payload)
        for key in list(WANDB_STEP_CACHE.keys()):
            WANDB_STEP_CACHE[key].clear()

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._flush_custom_metrics(state)

    def on_train_end(self, args, state, control, **kwargs):
        self._flush_custom_metrics(state, force=True)


def init_wandb(args: argparse.Namespace, train_size: int) -> Optional[str]:
    global WANDB_RUN
    if args.wandb_mode == "disabled" or wandb is None or not is_main_process():
        return None

    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["TEMPVQA_NUM_GENERATIONS"] = str(args.num_generations)

    run_id_file = Path(args.output_dir) / "wandb_run_id.txt"
    run_id = None
    if args.resume and run_id_file.exists():
        run_id = run_id_file.read_text(encoding="utf-8").strip() or None

    WANDB_RUN = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        id=run_id,
        resume="allow",
        config={
            "script_path": SCRIPT_PATH,
            "model_name_or_path": args.model_name_or_path,
            "train_dir": args.train_dir,
            "output_dir": args.output_dir,
            "cache_dir": args.cache_dir,
            "seed": args.seed,
            "num_frames": args.num_frames,
            "frame_width": args.frame_width,
            "frame_height": args.frame_height,
            "train_size": train_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "num_train_epochs": args.num_train_epochs,
            "save_steps": args.save_steps,
            "logging_steps": args.logging_steps,
            "save_total_limit": args.save_total_limit,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "generation_batch_size": args.generation_batch_size,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "reward_weight_consistency_deprecated": args.reward_weight_consistency,
            "reward_weight_answer_deprecated": args.reward_weight_answer,
            "reward_weight_json_alignment_deprecated": args.reward_weight_json_alignment,
            "reward_type": "grouped_q0_q1_q2_short_schema_curriculum",
            "reward_curriculum": "0-35: q0=.15 q1=.25 q2=.40 all=.20; 35-70: q0=.10 q1=.20 q2=.35 all=.35; 70-100: all=1.0",
            "training_unit": "one q0/q1/q2 group per rollout, exact short JSON schema",
            "grouped_steps_note": "default num_train_epochs=5 because grouped dataset has ~301 steps/epoch; this matches old ~1412-step single-question exposure",
            "beta": args.beta,
            "scale_rewards": args.scale_rewards,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
        },
    )
    if WANDB_RUN is not None and WANDB_RUN.id:
        run_id_file.write_text(WANDB_RUN.id, encoding="utf-8")
    return WANDB_RUN.id if WANDB_RUN is not None else None


def main() -> None:
    try:
        args = parse_args()

        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ["TEMPVQA_NUM_GENERATIONS"] = str(args.num_generations)

        train_dir = Path(args.train_dir)
        output_dir = Path(args.output_dir)
        cache_dir = Path(args.cache_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        processor = AutoProcessor.from_pretrained(args.model_name_or_path, use_fast=False)

        records = load_train_records(
            train_dir=train_dir,
            cache_dir=cache_dir,
            num_frames=args.num_frames,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
        )
        if args.max_examples is not None:
            records = records[: args.max_examples]

        if is_main_process():
            log(
                f"Grouped training has {len(records)} q0/q1/q2 examples. "
                "Each example contains 3 questions. "
                f"num_train_epochs={args.num_train_epochs}; total trainer steps will be computed by GRPO/Accelerate. "
                "If this is run on 2 GPUs with per_device_train_batch_size=1, expect roughly len(records)/2 steps per epoch."
            )

        train_ds = Dataset.from_list(records).with_transform(example_to_vlm_record)

        init_wandb(args, train_size=len(records))

        model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=model_dtype,
            device_map=None,
        )
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
        )

        grpo_kwargs = dict(
            output_dir=str(output_dir),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            beta=args.beta,
            scale_rewards=args.scale_rewards,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            max_grad_norm=args.max_grad_norm,
            remove_unused_columns=False,
            bf16=args.bf16,
            fp16=args.fp16 and not args.bf16,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
            report_to="wandb" if args.wandb_mode != "disabled" else "none",
            log_completions=False,
        )

        try:
            grpo_args = GRPOConfig(generation_batch_size=args.generation_batch_size, **grpo_kwargs)
        except TypeError:
            grpo_args = GRPOConfig(**grpo_kwargs)

        trainer = GRPOTrainer(
            model=model,
            processing_class=processor,
            reward_funcs=strict_json_reward,
            args=grpo_args,
            train_dataset=train_ds,
            peft_config=lora_cfg,
            callbacks=[EpochTrackerCallback(), WandbMetricsCallback()],
        )

        resume_ckpt = get_last_checkpoint(str(output_dir)) if args.resume else None
        trainer.train(resume_from_checkpoint=resume_ckpt)

        final_dir = output_dir / "final_adapter"
        trainer.save_model(str(final_dir))
        processor.save_pretrained(str(final_dir))

        with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

        if WANDB_RUN is not None and is_main_process():
            wandb.finish()

    except Exception:
        log_exception("main")
        raise


if __name__ == "__main__":
    main()