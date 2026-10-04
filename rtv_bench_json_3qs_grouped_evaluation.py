#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageOps
from peft import PeftConfig, PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


GROUP_ID_RE = re.compile(r"^q-group-([a-z0-9]+)-([012])(?:-[^-]+)*$", re.IGNORECASE)
MAIN_CATEGORIES = ("Object", "Action", "Event")
DIM_CATEGORIES = ("Perception", "Understanding", "Reasoning")
_DURATION_CACHE: Dict[str, float] = {}


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    prefix = f"[{now_str()}][host={socket.gethostname()}][pid={os.getpid()}]"
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def log_exception(context: str) -> None:
    log(f"EXCEPTION in {context}")
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grouped q0/q1/q2 short-schema TemporalVQA inference/evaluation."
    )
    p.add_argument("--test_dir", type=str, required=True, help="Directory containing QA.json and videos/")
    p.add_argument("--model_name_or_path", type=str, required=True, help="Base model path, or adapter path if adapter_config.json exists")
    p.add_argument("--adapter_path", type=str, default=None, help="Optional PEFT/LoRA adapter path")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda", help="cuda, cpu, or auto")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--frame_width", type=int, default=448)
    p.add_argument("--frame_height", type=int, default=448)
    p.add_argument("--image_mode", choices=["contact_sheet", "frames"], default="contact_sheet",
                   help="contact_sheet matches the grouped training script; frames matches the older eval style")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Limit number of complete q0/q1/q2 groups")
    p.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy generation")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
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
    if "what color" in q or "which color" in q or "what is the color" in q or "color of" in q:
        return "attribute"
    if "what is the emotional state" in q or "emotion" in q:
        return "event"
    if "purpose" in q or "want to do" in q or "why" in q:
        if "future" in q or "happen next" in q:
            return "future"
        return "intent"
    if "what will" in q or "about to" in q or "happen next" in q:
        return "future"
    if "did" in q or "doing" in q or "action" in q:
        return "action"
    if parse_dimension(type_str) == "Reasoning":
        return "reasoning"
    if parse_main_category(type_str) == "Event":
        return "event"
    return "other"


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text))


def load_annotations(data_dir: Path) -> List[Dict[str, Any]]:
    qa_path = data_dir / "QA.json"
    if not qa_path.exists():
        raise FileNotFoundError(f"Missing QA.json: {qa_path}")
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {qa_path}")
    return data


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
        prev_t: Optional[float] = None
        for item in sorted(group_items, key=sort_key):
            prev_map[item["questionID"]] = prev_t
            prev_t = float(item.get("end_time", 0.0))
    return prev_map


def build_option_prompt(options: Dict[str, str]) -> str:
    keys = sorted(options.keys())
    return "Choices:\n" + "\n".join([f"{k}. {options[k]}" for k in keys])


def single_question_block(item: Dict[str, Any], qlevel: str, previous_timestamp: Optional[float]) -> str:
    prev_ts = "null" if previous_timestamp is None else f"{float(previous_timestamp):.3f}"
    qtype_name = infer_question_type(item.get("question", ""), str(item.get("type", "")))
    return f"""
{qlevel.upper()}:
question_id: {item['questionID']}
queried_timestamp: {float(item['end_time']):.3f}
previous_queried_timestamp: {prev_ts}
expected_question_type: {qtype_name}
question: {item['question']}
{build_option_prompt(item['options'])}
""".strip()


def build_grouped_short_schema_prompt(
    q0_item: Dict[str, Any],
    q1_item: Dict[str, Any],
    q2_item: Dict[str, Any],
    prev_map: Dict[str, Optional[float]],
    latest_end_time: float,
) -> str:
    q0_block = single_question_block(q0_item, "q0", prev_map.get(q0_item["questionID"]))
    q1_block = single_question_block(q1_item, "q1", prev_map.get(q1_item["questionID"]))
    q2_block = single_question_block(q2_item, "q2", prev_map.get(q2_item["questionID"]))
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


def group_annotations(items: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    skipped_bad_group_id = 0
    for item in items:
        gid, qtype_num = parse_group_id_and_qtype(item.get("questionID", ""))
        if gid is None or qtype_num not in {"0", "1", "2"}:
            skipped_bad_group_id += 1
            continue
        grouped[gid][f"q{qtype_num}"] = item

    records: List[Dict[str, Any]] = []
    skipped_incomplete = 0
    for gid, qdict in sorted(grouped.items()):
        if not {"q0", "q1", "q2"}.issubset(qdict.keys()):
            skipped_incomplete += 1
            continue
        q0, q1, q2 = qdict["q0"], qdict["q1"], qdict["q2"]
        latest_end_time = max(float(q0.get("end_time", 0.0)), float(q1.get("end_time", 0.0)), float(q2.get("end_time", 0.0)))
        records.append({"group_id": gid, "q0": q0, "q1": q1, "q2": q2, "latest_end_time": latest_end_time})
        if limit is not None and len(records) >= limit:
            break

    log(f"Built {len(records)} complete q0/q1/q2 groups (skipped_incomplete={skipped_incomplete}, skipped_bad_group_id={skipped_bad_group_id})")
    return records


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


def validate_grouped_schema(obj: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, bool], List[str]]:
    errors: List[str] = []
    per_q_schema_valid: Dict[str, bool] = {"q0": False, "q1": False, "q2": False}
    required_top = {"q0", "q1", "q2"}
    required_sub = {"final_answer", "rationale"}

    if not isinstance(obj, dict):
        return False, per_q_schema_valid, ["not_a_dict"]

    top_keys = set(obj.keys())
    for key in sorted(required_top - top_keys):
        errors.append(f"missing_top:{key}")
    for key in sorted(top_keys - required_top):
        errors.append(f"extra_top:{key}")

    for qk in ("q0", "q1", "q2"):
        sub = obj.get(qk)
        q_errors: List[str] = []
        if not isinstance(sub, dict):
            q_errors.append("not_a_dict")
        else:
            sub_keys = set(sub.keys())
            for key in sorted(required_sub - sub_keys):
                q_errors.append(f"missing:{key}")
            for key in sorted(sub_keys - required_sub):
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


def extract_grouped_pred_letters(parsed: Optional[Dict[str, Any]], q0_options: Dict[str, str], q1_options: Dict[str, str], q2_options: Dict[str, str]) -> Dict[str, str]:
    preds = {"q0": "ERROR", "q1": "ERROR", "q2": "ERROR"}
    if not isinstance(parsed, dict):
        return preds
    for qk, opts in (("q0", q0_options), ("q1", q1_options), ("q2", q2_options)):
        sub = parsed.get(qk)
        if isinstance(sub, dict):
            fa = sub.get("final_answer")
            if isinstance(fa, str) and fa.strip().upper() in opts:
                preds[qk] = fa.strip().upper()
    return preds


def get_video_duration(video_path: str) -> float:
    if video_path in _DURATION_CACHE:
        return _DURATION_CACHE[video_path]
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
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


def extract_frame(video_path: str, ts: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration(video_path)
    safe_ts = min(max(0.0, ts), max(0.0, duration - 0.25))
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp.{os.getpid()}{out_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    tried: List[float] = []
    for candidate_ts in [safe_ts, max(0.0, safe_ts - 1.0), max(0.0, safe_ts - 2.0)]:
        if any(abs(candidate_ts - old) < 1e-6 for old in tried):
            continue
        tried.append(candidate_ts)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{candidate_ts:.3f}", "-i", video_path, "-frames:v", "1", "-q:v", "2", tmp_path.as_posix()]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            os.replace(tmp_path, out_path)
            return
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    raise RuntimeError(f"Failed to extract frame for {video_path} at ts={ts:.3f}, tried={tried}")


def ensure_frames(src_video: str, end_time: float, frame_dir: Path, num_frames: int) -> Tuple[List[str], List[float]]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration(src_video)
    timestamps = sample_timestamps(end_time, num_frames, video_duration=duration)
    frame_paths: List[str] = []
    for i, ts in enumerate(timestamps):
        fp = frame_dir / f"frame_{i:02d}.jpg"
        if not fp.exists() or fp.stat().st_size == 0:
            extract_frame(src_video, ts, fp)
        frame_paths.append(fp.as_posix())
    return frame_paths, timestamps


def ensure_contact_sheet(src_video: str, end_time: float, sheet_path: Path, num_frames: int, width: int, height: int) -> str:
    if sheet_path.exists() and sheet_path.stat().st_size > 0:
        return sheet_path.as_posix()
    frame_dir = sheet_path.parent / f"{sheet_path.stem}_frames"
    frame_paths, timestamps = ensure_frames(src_video, end_time, frame_dir, num_frames)
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
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sheet_path.with_name(f"{sheet_path.stem}.tmp.{os.getpid()}{sheet_path.suffix}")
    sheet.save(tmp, quality=95)
    os.replace(tmp, sheet_path)
    return sheet_path.as_posix()


class Counter:
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def add(self, is_correct: bool, n: int = 1) -> None:
        self.total += n
        self.correct += int(bool(is_correct)) * n

    @property
    def acc(self) -> Optional[float]:
        return (self.correct / self.total) if self.total else None


def resolve_model_and_adapter(model_name_or_path: str, adapter_path: Optional[str]) -> Tuple[str, Optional[str]]:
    if adapter_path:
        return model_name_or_path, adapter_path
    maybe_adapter = Path(model_name_or_path)
    if maybe_adapter.exists() and (maybe_adapter / "adapter_config.json").exists():
        peft_cfg = PeftConfig.from_pretrained(model_name_or_path)
        return peft_cfg.base_model_name_or_path, model_name_or_path
    return model_name_or_path, None


class QwenGroupedShortSchemaRunner:
    def __init__(self, model_name_or_path: str, adapter_path: Optional[str], dtype: str = "auto", device: str = "cuda"):
        base_model_path, adapter_to_load = resolve_model_and_adapter(model_name_or_path, adapter_path)
        self.base_model_path = base_model_path
        self.adapter_path = adapter_to_load

        if dtype == "auto":
            model_dtype = "auto"
        else:
            model_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

        use_device_map = device == "auto"
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            dtype=model_dtype,
            device_map="auto" if use_device_map else None,
        )
        if adapter_to_load is not None:
            log(f"Loading PEFT adapter from: {adapter_to_load}")
            self.model = PeftModel.from_pretrained(base_model, adapter_to_load)
        else:
            self.model = base_model

        processor_source = adapter_to_load if (adapter_to_load and (Path(adapter_to_load) / "preprocessor_config.json").exists()) else base_model_path
        self.processor = AutoProcessor.from_pretrained(processor_source, use_fast=False)

        if not use_device_map:
            self.device = device
            self.model.to(device)
        else:
            self.device = "auto"
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[Dict[str, str], str, Optional[Dict[str, Any]], bool, Dict[str, bool], List[str]]:
        content = [{"type": "image", "image": p} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image_paths], padding=True, return_tensors="pt")
        if self.device != "auto":
            target_device = torch.device(self.device)
            inputs = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        out_ids = generated_ids[:, input_len:]
        raw_text = self.processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        parsed_json = try_parse_json(raw_text)
        schema_valid, per_q_schema_valid, schema_errors = validate_grouped_schema(parsed_json)
        # Options are checked outside where the group item is available; use A-D here for schema only.
        preds = {"q0": "ERROR", "q1": "ERROR", "q2": "ERROR"}
        if isinstance(parsed_json, dict):
            for qk in ("q0", "q1", "q2"):
                sub = parsed_json.get(qk)
                if isinstance(sub, dict):
                    fa = sub.get("final_answer")
                    if isinstance(fa, str) and fa.strip().upper() in {"A", "B", "C", "D"}:
                        preds[qk] = fa.strip().upper()
        return preds, raw_text, parsed_json, schema_valid, per_q_schema_valid, schema_errors


def load_existing_group_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return {}
    return {item["group_id"]: item for item in data if isinstance(item, dict) and "group_id" in item}


def fmt_rate(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{100.0 * x:.2f}%"


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def compute_reports(flat_rows: List[Dict[str, Any]], grouped_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = Counter()
    qtype_stats = defaultdict(Counter)
    main_stats = defaultdict(Counter)
    dim_stats = defaultdict(Counter)
    json_valid = Counter()
    schema_valid = Counter()
    all_correct = Counter()
    parsed_all = Counter()

    for row in flat_rows:
        correct = bool(row.get("correct", False))
        overall.add(correct)
        qtype = str(row.get("qtype", ""))
        if qtype in {"0", "1", "2"}:
            qtype_stats[qtype].add(correct)
        main_cat = parse_main_category(row.get("type"))
        if main_cat is not None:
            main_stats[main_cat].add(correct)
        dim_cat = parse_dimension(row.get("type"))
        if dim_cat is not None:
            dim_stats[dim_cat].add(correct)

    q2_score_correct = 0
    q2_score_total = 0
    for grow in grouped_rows:
        json_valid.add(bool(grow.get("json_valid", False)))
        schema_valid.add(bool(grow.get("schema_valid", False)))
        parsed_all.add(bool(grow.get("parsed_all_answers", False)))
        all_correct.add(bool(grow.get("all_correct", False)))
        q2_score_total += 1
        if bool(grow.get("q0_correct", False)) and bool(grow.get("q1_correct", False)) and bool(grow.get("q2_correct", False)):
            q2_score_correct += 1

    return {
        "num_groups": len(grouped_rows),
        "num_items": len(flat_rows),
        "accuracy": overall.acc,
        "accuracy_num": overall.correct,
        "accuracy_den": overall.total,
        "score": (q2_score_correct / q2_score_total) if q2_score_total else None,
        "score_num": q2_score_correct,
        "score_den": q2_score_total,
        "all_q0_q1_q2_correct_rate": all_correct.acc,
        "all_q0_q1_q2_correct_num": all_correct.correct,
        "all_q0_q1_q2_correct_den": all_correct.total,
        "qtype_accuracy": {
            q: {"correct": qtype_stats[q].correct, "total": qtype_stats[q].total, "acc": qtype_stats[q].acc}
            for q in ("0", "1", "2") if q in qtype_stats
        },
        "main_category_accuracy": {
            k: {"correct": main_stats[k].correct, "total": main_stats[k].total, "acc": main_stats[k].acc}
            for k in MAIN_CATEGORIES if k in main_stats
        },
        "dimension_accuracy": {
            k: {"correct": dim_stats[k].correct, "total": dim_stats[k].total, "acc": dim_stats[k].acc}
            for k in DIM_CATEGORIES if k in dim_stats
        },
        "json_valid_rate": json_valid.acc,
        "json_valid_num": json_valid.correct,
        "json_valid_den": json_valid.total,
        "schema_valid_rate": schema_valid.acc,
        "schema_valid_num": schema_valid.correct,
        "schema_valid_den": schema_valid.total,
        "parsed_all_answers_rate": parsed_all.acc,
        "parsed_all_answers_num": parsed_all.correct,
        "parsed_all_answers_den": parsed_all.total,
    }


def main() -> None:
    args = parse_args()
    try:
        test_dir = Path(args.test_dir).resolve()
        videos_dir = test_dir / "videos"
        if not videos_dir.exists():
            raise FileNotFoundError(f"Missing videos dir: {videos_dir}")
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (output_dir / "frame_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        grouped_preds_path = output_dir / "grouped_predictions.json"
        flat_preds_path = output_dir / "predictions_flat.json"
        summary_path = output_dir / "summary.json"

        annotations = load_annotations(test_dir)
        prev_map = build_previous_timestamp_map(annotations)
        groups = group_annotations(annotations, limit=args.limit)
        log(f"Loaded {len(annotations)} raw test samples from {test_dir / 'QA.json'}")

        existing = load_existing_group_predictions(grouped_preds_path) if args.resume else {}
        if existing:
            log(f"Resuming from existing grouped predictions: {len(existing)} groups")

        runner = QwenGroupedShortSchemaRunner(args.model_name_or_path, args.adapter_path, dtype=args.dtype, device=args.device)
        log(f"Base model: {runner.base_model_path}")
        log(f"Adapter:    {runner.adapter_path}")
        log(f"Image mode: {args.image_mode}")

        grouped_results: List[Dict[str, Any]] = []

        for idx, group in enumerate(groups, start=1):
            gid = group["group_id"]
            if gid in existing and isinstance(existing[gid].get("preds"), dict):
                grouped_results.append(existing[gid])
                continue

            q0, q1, q2 = group["q0"], group["q1"], group["q2"]
            video_name = q2["video"]
            src_video = (videos_dir / video_name).resolve()
            if not src_video.exists():
                raise FileNotFoundError(f"Video not found: {videos_dir / video_name}")
            latest_end_time = float(group["latest_end_time"])
            prompt = build_grouped_short_schema_prompt(q0, q1, q2, prev_map, latest_end_time)

            raw_output = ""
            parsed_json = None
            schema_valid = False
            per_q_schema_valid = {"q0": False, "q1": False, "q2": False}
            schema_errors: List[str] = []
            preds = {"q0": "ERROR", "q1": "ERROR", "q2": "ERROR"}
            error = None
            image_paths: List[str] = []

            try:
                if args.image_mode == "contact_sheet":
                    sheet_path = cache_dir / "grouped_contact_sheets" / f"group_{safe_slug(gid)}_{latest_end_time:.3f}.jpg"
                    image_paths = [ensure_contact_sheet(str(src_video), latest_end_time, sheet_path, args.num_frames, args.frame_width, args.frame_height)]
                else:
                    frame_dir = cache_dir / "grouped_frames" / f"group_{safe_slug(gid)}_{latest_end_time:.3f}_frames"
                    image_paths, _ = ensure_frames(str(src_video), latest_end_time, frame_dir, args.num_frames)

                preds, raw_output, parsed_json, schema_valid, per_q_schema_valid, schema_errors = runner.predict(
                    image_paths=image_paths,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                # Re-extract with real option sets so invalid letters/options are caught consistently.
                preds = extract_grouped_pred_letters(parsed_json, q0["options"], q1["options"], q2["options"])
            except Exception as e:
                error = str(e)
                log(f"[{idx}/{len(groups)}] ERROR group={gid}: {error}")

            q0_gold = str(q0.get("answer", "")).strip().upper()
            q1_gold = str(q1.get("answer", "")).strip().upper()
            q2_gold = str(q2.get("answer", "")).strip().upper()
            q0_correct = preds["q0"] == q0_gold
            q1_correct = preds["q1"] == q1_gold
            q2_correct = preds["q2"] == q2_gold
            all_correct = q0_correct and q1_correct and q2_correct
            parsed_all_answers = all(preds[qk] != "ERROR" for qk in ("q0", "q1", "q2"))

            row = {
                "group_id": gid,
                "video": video_name,
                "latest_end_time": latest_end_time,
                "image_mode": args.image_mode,
                "image_paths": image_paths,
                "prompt": prompt,
                "raw_output": raw_output,
                "parsed_json": parsed_json,
                "json_valid": parsed_json is not None,
                "schema_valid": schema_valid,
                "per_q_schema_valid": per_q_schema_valid,
                "schema_errors": schema_errors,
                "preds": preds,
                "golds": {"q0": q0_gold, "q1": q1_gold, "q2": q2_gold},
                "q0_correct": q0_correct,
                "q1_correct": q1_correct,
                "q2_correct": q2_correct,
                "all_correct": all_correct,
                "parsed_all_answers": parsed_all_answers,
                "q0_questionID": q0["questionID"],
                "q1_questionID": q1["questionID"],
                "q2_questionID": q2["questionID"],
                "q0_item": q0,
                "q1_item": q1,
                "q2_item": q2,
                "model_name": runner.base_model_path,
                "adapter_path": runner.adapter_path,
                "method": "grouped_short_schema_q0_q1_q2_json",
            }
            if error is not None:
                row["error"] = error
            grouped_results.append(row)

            if idx % 10 == 0 or idx == len(groups):
                save_json(grouped_preds_path, grouped_results)
                running_all = sum(int(x.get("all_correct", False)) for x in grouped_results)
                running_schema = sum(int(x.get("schema_valid", False)) for x in grouped_results)
                log(f"[{idx}/{len(groups)}] saved | all_correct={running_all}/{len(grouped_results)} | schema_valid={running_schema}/{len(grouped_results)}")

        # Build flat rows so old per-question reports remain easy to compare.
        flat_rows: List[Dict[str, Any]] = []
        for grow in grouped_results:
            for qk, qnum in (("q0", "0"), ("q1", "1"), ("q2", "2")):
                sample = grow[f"{qk}_item"]
                pred = grow["preds"].get(qk, "ERROR")
                gold = grow["golds"].get(qk, str(sample.get("answer", "")).strip().upper())
                row = dict(sample)
                row.update({
                    "group_id": grow["group_id"],
                    "qtype": qnum,
                    "pred": pred,
                    "gold": gold,
                    "raw_output": grow["raw_output"],
                    "parsed_json": grow["parsed_json"],
                    "json_valid": grow["json_valid"],
                    "schema_valid": bool(grow.get("per_q_schema_valid", {}).get(qk, False)),
                    "group_schema_valid": grow["schema_valid"],
                    "correct": pred == gold,
                    "all_group_correct": grow["all_correct"],
                    "model_name": grow["model_name"],
                    "adapter_path": grow["adapter_path"],
                    "method": grow["method"],
                })
                flat_rows.append(row)

        save_json(grouped_preds_path, grouped_results)
        save_json(flat_preds_path, flat_rows)
        summary = compute_reports(flat_rows, grouped_results)
        save_json(summary_path, summary)

        log(f"Grouped predictions saved to: {grouped_preds_path}")
        log(f"Flat predictions saved to:    {flat_preds_path}")
        log(f"Summary saved to:             {summary_path}")
        log(f"Accuracy:     {summary['accuracy_num']}/{summary['accuracy_den']} = {fmt_rate(summary['accuracy'])}")
        log(f"All-correct:  {summary['all_q0_q1_q2_correct_num']}/{summary['all_q0_q1_q2_correct_den']} = {fmt_rate(summary['all_q0_q1_q2_correct_rate'])}")
        log(f"Score:        {summary['score_num']}/{summary['score_den']} = {fmt_rate(summary['score'])}")
        log(f"JSON valid:   {summary['json_valid_num']}/{summary['json_valid_den']} = {fmt_rate(summary['json_valid_rate'])}")
        log(f"Schema valid: {summary['schema_valid_num']}/{summary['schema_valid_den']} = {fmt_rate(summary['schema_valid_rate'])}")

        print("=" * 80)
        print(f"base model:   {runner.base_model_path}")
        print(f"adapter:      {runner.adapter_path}")
        print("method:       grouped_short_schema_q0_q1_q2_json")
        print(f"image_mode:   {args.image_mode}")
        print(f"test_dir:     {test_dir}")
        print(f"groups:       {summary['num_groups']}")
        print(f"items:        {summary['num_items']}")
        print(f"accuracy:     {summary['accuracy_num']}/{summary['accuracy_den']} = {fmt_rate(summary['accuracy'])}")
        print(f"all-correct:  {summary['all_q0_q1_q2_correct_num']}/{summary['all_q0_q1_q2_correct_den']} = {fmt_rate(summary['all_q0_q1_q2_correct_rate'])}")
        print(f"score:        {summary['score_num']}/{summary['score_den']} = {fmt_rate(summary['score'])}")
        print(f"json valid:   {summary['json_valid_num']}/{summary['json_valid_den']} = {fmt_rate(summary['json_valid_rate'])}")
        print(f"schema valid: {summary['schema_valid_num']}/{summary['schema_valid_den']} = {fmt_rate(summary['schema_valid_rate'])}")
        if summary["qtype_accuracy"]:
            print("qtype:")
            for q, stats in summary["qtype_accuracy"].items():
                print(f"  q{q}: {stats['correct']}/{stats['total']} = {fmt_rate(stats['acc'])}")
        if summary["main_category_accuracy"]:
            print("main categories:")
            for k, stats in summary["main_category_accuracy"].items():
                print(f"  {k}: {stats['correct']}/{stats['total']} = {fmt_rate(stats['acc'])}")
        if summary["dimension_accuracy"]:
            print("dimensions:")
            for k, stats in summary["dimension_accuracy"].items():
                print(f"  {k}: {stats['correct']}/{stats['total']} = {fmt_rate(stats['acc'])}")

    except Exception:
        log_exception("main")
        raise


if __name__ == "__main__":
    main()