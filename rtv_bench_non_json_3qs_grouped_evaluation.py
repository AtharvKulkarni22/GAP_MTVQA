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
LETTER_SET = {"A", "B", "C", "D"}
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
        description=(
            "Grouped q0/q1/q2 RTV-Bench/TemporalVQA evaluation without JSON/schema output. "
            "The model sees the same grouped QA setting as grouped training, but answers freely."
        )
    )
    p.add_argument("--test_dir", type=str, required=True, help="Directory with QA.json and videos/")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--adapter_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--image_mode", choices=["contact_sheet", "frames"], default="contact_sheet")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit_groups", type=int, default=None)
    p.add_argument("--limit_items", type=int, default=None, help="Optional item limit before grouping, mainly for debugging")
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


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


def load_annotations(test_dir: Path) -> List[Dict[str, Any]]:
    qa_path = test_dir / "QA.json"
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
        group_items = sorted(group_items, key=sort_key)
        prev_t: Optional[float] = None
        for item in group_items:
            prev_map[item["questionID"]] = prev_t
            prev_t = float(item.get("end_time", 0.0))
    return prev_map


def group_annotations(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prev_map = build_previous_timestamp_map(items)
    grouped_items: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    skipped_bad_group_id = 0

    for item in items:
        gid, qtype_num = parse_group_id_and_qtype(item.get("questionID", ""))
        if gid is None or qtype_num not in {"0", "1", "2"}:
            skipped_bad_group_id += 1
            continue
        grouped_items[gid][f"q{qtype_num}"] = item

    groups: List[Dict[str, Any]] = []
    skipped_incomplete = 0
    for gid, qdict in sorted(grouped_items.items()):
        if not {"q0", "q1", "q2"}.issubset(qdict.keys()):
            skipped_incomplete += 1
            continue
        q0_item, q1_item, q2_item = qdict["q0"], qdict["q1"], qdict["q2"]
        latest_end_time = max(
            float(q0_item.get("end_time", 0.0)),
            float(q1_item.get("end_time", 0.0)),
            float(q2_item.get("end_time", 0.0)),
        )
        groups.append(
            {
                "group_id": gid,
                "video": q2_item["video"],
                "latest_end_time": latest_end_time,
                "q0": q0_item,
                "q1": q1_item,
                "q2": q2_item,
                "q0_previous_timestamp": prev_map.get(q0_item["questionID"]),
                "q1_previous_timestamp": prev_map.get(q1_item["questionID"]),
                "q2_previous_timestamp": prev_map.get(q2_item["questionID"]),
                "q0_question_type": infer_question_type(q0_item["question"], str(q0_item.get("type", ""))),
                "q1_question_type": infer_question_type(q1_item["question"], str(q1_item.get("type", ""))),
                "q2_question_type": infer_question_type(q2_item["question"], str(q2_item.get("type", ""))),
            }
        )

    log(
        f"Grouped annotations: {len(groups)} complete groups "
        f"(skipped_incomplete={skipped_incomplete}, skipped_bad_group_id={skipped_bad_group_id})"
    )
    return groups


def build_option_prompt(options: Dict[str, str]) -> str:
    keys = sorted(options.keys())
    return "Choices:\n" + "\n".join([f"{k}. {options[k]}" for k in keys])


def _single_question_block_for_group(
    item: Dict[str, Any],
    qlevel: str,
    question_type: str,
    previous_timestamp: Optional[float],
) -> str:
    prev_ts = "none" if previous_timestamp is None else f"{float(previous_timestamp):.3f}"
    return f"""
{qlevel.upper()}:
question_id: {item['questionID']}
queried_timestamp: {float(item['end_time']):.3f}
previous_queried_timestamp: {prev_ts}
expected_question_type: {question_type}
question: {item['question']}
{build_option_prompt(item['options'])}
""".strip()


def build_grouped_no_json_prompt(group: Dict[str, Any]) -> str:
    q0_block = _single_question_block_for_group(group["q0"], "q0", group["q0_question_type"], group["q0_previous_timestamp"])
    q1_block = _single_question_block_for_group(group["q1"], "q1", group["q1_question_type"], group["q1_previous_timestamp"])
    q2_block = _single_question_block_for_group(group["q2"], "q2", group["q2_question_type"], group["q2_previous_timestamp"])
    latest_end_time = float(group["latest_end_time"])

    return f"""
You are a careful temporal video reasoning assistant.

You are given video frames sampled in chronological order from time 0 up to {latest_end_time:.3f} seconds.
The frame labels show approximate timestamps. Answer each question using the scene state at that question's own queried_timestamp.

Answer all three related hierarchy questions q0, q1, and q2 for the same temporal group.
Do not use JSON, XML, markdown tables, or code fences.
You may briefly reason in natural language, but the final line must be exactly in this format:
Final answers: q0=A, q1=B, q2=C

Questions:
{q0_block}

{q1_block}

{q2_block}

Rules:
1. For each question, choose exactly one option letter from that question's choices: A, B, C, or D.
2. q0, q1, and q2 may refer to different timestamps; use each question's own queried_timestamp.
3. Keep the explanation short.
4. The final line must contain all three option letters as: Final answers: q0=<letter>, q1=<letter>, q2=<letter>
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
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{candidate_ts:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            tmp_path.as_posix(),
        ]
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


def make_contact_sheet(frame_paths: List[str], timestamps: List[float], out_path: Path, width: int = 448, height: int = 448) -> str:
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path.as_posix()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = len(frame_paths)
    cols = min(4, max(1, num_frames))
    rows = math.ceil(num_frames / cols)
    sheet = Image.new("RGB", (cols * width, rows * height), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, (fp, ts) in enumerate(zip(frame_paths, timestamps)):
        with Image.open(fp) as img:
            img = img.convert("RGB")
            img = ImageOps.fit(img, (width, height))
            x = (idx % cols) * width
            y = (idx // cols) * height
            sheet.paste(img, (x, y))
            draw.rectangle([(x, y), (x + 110, y + 28)], fill=(255, 255, 255))
            draw.text((x + 6, y + 6), f"t={ts:.1f}s", fill=(0, 0, 0))
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp.{os.getpid()}{out_path.suffix}")
    sheet.save(tmp_path, quality=95)
    os.replace(tmp_path, out_path)
    return out_path.as_posix()


def extract_json_substring(text: str) -> Optional[str]:
    # Used only as a diagnostic. The model is not asked for JSON here.
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    return text[start:end + 1]


def has_json_object(text: str) -> bool:
    payload = extract_json_substring(text)
    if payload is None:
        return False
    try:
        obj = json.loads(payload)
        return isinstance(obj, dict)
    except Exception:
        return False


def _normalize_letter(x: str) -> str:
    x = (x or "").strip().upper()
    return x if x in LETTER_SET else "ERROR"


def extract_grouped_answers_no_json(text: str) -> Dict[str, str]:
    """Robustly parse q0/q1/q2 answer letters from free-form text."""
    raw = text or ""
    preds = {"q0": "ERROR", "q1": "ERROR", "q2": "ERROR"}

    # Preferred pattern: Final answers: q0=A, q1=B, q2=C
    for qk in ("q0", "q1", "q2"):
        m = re.search(rf"\b{qk}\b\s*[:=\-]\s*\(?\s*([A-D])\s*\)?", raw, flags=re.IGNORECASE)
        if m:
            preds[qk] = _normalize_letter(m.group(1))

    if all(v != "ERROR" for v in preds.values()):
        return preds

    # Common line variants: "Q0: A", "q1 answer is B", "For q2, C"
    for qk in ("q0", "q1", "q2"):
        if preds[qk] != "ERROR":
            continue
        patterns = [
            rf"\b{qk}\b[^A-Da-d\n]{{0,80}}\banswer\b[^A-Da-d\n]{{0,30}}\b([A-D])\b",
            rf"\bfor\s+{qk}\b[^A-Da-d\n]{{0,80}}\b([A-D])\b",
            rf"\b{qk.upper()}\b[^A-Da-d\n]{{0,80}}\b([A-D])\b",
        ]
        for pat in patterns:
            m = re.search(pat, raw, flags=re.IGNORECASE)
            if m:
                preds[qk] = _normalize_letter(m.group(1))
                break

    if all(v != "ERROR" for v in preds.values()):
        return preds

    # Last resort: if a final line contains exactly three standalone letters, map them in order.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in reversed(lines[-5:]):
        letters = re.findall(r"\b([A-D])\b", ln, flags=re.IGNORECASE)
        if len(letters) >= 3:
            ordered = [_normalize_letter(x) for x in letters[:3]]
            for qk, val in zip(("q0", "q1", "q2"), ordered):
                if preds[qk] == "ERROR":
                    preds[qk] = val
            break

    return preds


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


class QwenGroupedNoJSONRunner:
    def __init__(self, model_name_or_path: str, adapter_path: Optional[str], dtype: str = "auto", device: str = "cuda"):
        base_model_path, adapter_to_load = resolve_model_and_adapter(model_name_or_path, adapter_path)
        self.base_model_path = base_model_path
        self.adapter_path = adapter_to_load

        if dtype == "auto":
            model_dtype: Any = "auto"
        else:
            model_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[dtype]

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
        max_new_tokens: int = 256,
    ) -> Tuple[Dict[str, str], str, bool]:
        content = [{"type": "image", "image": p} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image_paths],
            padding=True,
            return_tensors="pt",
        )

        if self.device != "auto":
            target_device = torch.device(self.device)
            inputs = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        input_len = inputs["input_ids"].shape[1]
        out_ids = generated_ids[:, input_len:]
        raw_text = self.processor.batch_decode(
            out_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        preds = extract_grouped_answers_no_json(raw_text)
        used_json = has_json_object(raw_text)
        return preds, raw_text, used_json


def load_existing_grouped_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return {}
    return {item["group_id"]: item for item in data if isinstance(item, dict) and "group_id" in item}


def compute_reports(flat_data: List[Dict[str, Any]], grouped_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = Counter()
    qtype_stats = defaultdict(Counter)
    main_stats = defaultdict(Counter)
    dim_stats = defaultdict(Counter)
    parsed_answer = Counter()
    no_json_output = Counter()

    for item in flat_data:
        is_correct = bool(item.get("correct", False))
        overall.add(is_correct)
        parsed_answer.add(item.get("pred") in LETTER_SET)

        gid, qtype = parse_group_id_and_qtype(item.get("questionID"))
        if qtype is not None:
            qtype_stats[qtype].add(is_correct)
        main_cat = parse_main_category(item.get("type"))
        if main_cat is not None:
            main_stats[main_cat].add(is_correct)
        dim_cat = parse_dimension(item.get("type"))
        if dim_cat is not None:
            dim_stats[dim_cat].add(is_correct)

    all_correct = Counter()
    for group in grouped_data:
        all_correct.add(bool(group.get("all_correct", False)))
        no_json_output.add(not bool(group.get("used_json", False)))

    return {
        "num_groups": len(grouped_data),
        "num_items": len(flat_data),
        "accuracy": overall.acc,
        "accuracy_num": overall.correct,
        "accuracy_den": overall.total,
        "all_q0_q1_q2_correct_rate": all_correct.acc,
        "all_q0_q1_q2_correct_num": all_correct.correct,
        "all_q0_q1_q2_correct_den": all_correct.total,
        "score": all_correct.acc,
        "score_num": all_correct.correct,
        "score_den": all_correct.total,
        "parsed_answer_rate": parsed_answer.acc,
        "parsed_answer_num": parsed_answer.correct,
        "parsed_answer_den": parsed_answer.total,
        "no_json_output_rate": no_json_output.acc,
        "no_json_output_num": no_json_output.correct,
        "no_json_output_den": no_json_output.total,
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
    }


def fmt_rate(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{100.0 * x:.2f}%"


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_flat_rows(group_row: Dict[str, Any], group: Dict[str, Any]) -> List[Dict[str, Any]]:
    preds = group_row.get("preds", {})
    rows: List[Dict[str, Any]] = []
    for qk in ("q0", "q1", "q2"):
        sample = dict(group[qk])
        gold = str(sample.get("answer", "")).strip().upper()
        pred = preds.get(qk, "ERROR")
        sample.update(
            {
                "group_id": group["group_id"],
                "pred": pred,
                "correct": bool(pred == gold),
                "raw_output": group_row.get("raw_output", ""),
                "parsed_answer": pred in LETTER_SET,
                "used_json": group_row.get("used_json", False),
                "model_name": group_row.get("model_name"),
                "adapter_path": group_row.get("adapter_path"),
                "method": "grouped_no_json_q0_q1_q2",
            }
        )
        rows.append(sample)
    return rows


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
        if args.limit_items is not None:
            annotations = annotations[: args.limit_items]
        groups = group_annotations(annotations)
        if args.limit_groups is not None:
            groups = groups[: args.limit_groups]
        log(f"Loaded {len(groups)} complete q0/q1/q2 groups from {test_dir / 'QA.json'}")

        existing = load_existing_grouped_predictions(grouped_preds_path) if args.resume else {}
        if existing:
            log(f"Resuming from existing grouped predictions: {len(existing)} groups")

        runner = QwenGroupedNoJSONRunner(
            model_name_or_path=args.model_name_or_path,
            adapter_path=args.adapter_path,
            dtype=args.dtype,
            device=args.device,
        )
        log(f"Base model: {runner.base_model_path}")
        log(f"Adapter:    {runner.adapter_path}")

        grouped_results: List[Dict[str, Any]] = []
        flat_results: List[Dict[str, Any]] = []

        for idx, group in enumerate(groups, start=1):
            gid = group["group_id"]
            if gid in existing and isinstance(existing[gid].get("preds"), dict):
                row = existing[gid]
                grouped_results.append(row)
                flat_results.extend(build_flat_rows(row, group))
                continue

            video_name = group["video"]
            src_video = (videos_dir / video_name).resolve()
            if not src_video.exists():
                raise FileNotFoundError(f"Video not found: {videos_dir / video_name}")

            frame_dir = cache_dir / f"group_{safe_slug(gid)}_{float(group['latest_end_time']):.3f}_frames"
            prompt = build_grouped_no_json_prompt(group)
            preds = {"q0": "ERROR", "q1": "ERROR", "q2": "ERROR"}
            raw_output = ""
            used_json = False
            error = None
            image_paths_for_record: List[str] = []

            try:
                frame_paths, timestamps = ensure_frames(str(src_video), float(group["latest_end_time"]), frame_dir, args.num_frames)
                if args.image_mode == "contact_sheet":
                    sheet_path = make_contact_sheet(
                        frame_paths,
                        timestamps,
                        cache_dir / "contact_sheets" / f"group_{safe_slug(gid)}_{float(group['latest_end_time']):.3f}.jpg",
                    )
                    image_paths = [sheet_path]
                else:
                    image_paths = frame_paths
                image_paths_for_record = image_paths
                preds, raw_output, used_json = runner.predict(
                    image_paths=image_paths,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                error = str(e)
                log(f"[{idx}/{len(groups)}] ERROR group_id={gid}: {error}")

            q0_gold = str(group["q0"].get("answer", "")).strip().upper()
            q1_gold = str(group["q1"].get("answer", "")).strip().upper()
            q2_gold = str(group["q2"].get("answer", "")).strip().upper()
            q_correct = {
                "q0": bool(preds.get("q0") == q0_gold),
                "q1": bool(preds.get("q1") == q1_gold),
                "q2": bool(preds.get("q2") == q2_gold),
            }
            all_correct = all(q_correct.values())

            row = {
                "group_id": gid,
                "video": video_name,
                "latest_end_time": float(group["latest_end_time"]),
                "image_mode": args.image_mode,
                "image_paths": image_paths_for_record,
                "num_frames": args.num_frames,
                "prompt": prompt,
                "preds": preds,
                "gold": {"q0": q0_gold, "q1": q1_gold, "q2": q2_gold},
                "correct": q_correct,
                "all_correct": all_correct,
                "raw_output": raw_output,
                "used_json": used_json,
                "parsed_all_answers": all(preds.get(qk) in LETTER_SET for qk in ("q0", "q1", "q2")),
                "model_name": runner.base_model_path,
                "adapter_path": runner.adapter_path,
                "method": "grouped_no_json_q0_q1_q2",
                "q0_questionID": group["q0"]["questionID"],
                "q1_questionID": group["q1"]["questionID"],
                "q2_questionID": group["q2"]["questionID"],
            }
            if error is not None:
                row["error"] = error

            grouped_results.append(row)
            flat_results.extend(build_flat_rows(row, group))

            if idx % 10 == 0 or idx == len(groups):
                save_json(grouped_preds_path, grouped_results)
                save_json(flat_preds_path, flat_results)
                log(
                    f"[{idx}/{len(groups)}] saved | "
                    f"flat_acc={sum(int(x.get('correct', False)) for x in flat_results)}/{len(flat_results)} | "
                    f"all_correct={sum(int(x.get('all_correct', False)) for x in grouped_results)}/{len(grouped_results)} | "
                    f"parsed_all={sum(int(x.get('parsed_all_answers', False)) for x in grouped_results)}/{len(grouped_results)} | "
                    f"no_json={sum(int(not x.get('used_json', False)) for x in grouped_results)}/{len(grouped_results)}"
                )

        save_json(grouped_preds_path, grouped_results)
        save_json(flat_preds_path, flat_results)
        summary = compute_reports(flat_results, grouped_results)
        save_json(summary_path, summary)

        log(f"Grouped predictions saved to: {grouped_preds_path}")
        log(f"Flat predictions saved to:    {flat_preds_path}")
        log(f"Summary saved to:             {summary_path}")
        log(f"Accuracy:     {summary['accuracy_num']}/{summary['accuracy_den']} = {fmt_rate(summary['accuracy'])}")
        log(f"All-correct:  {summary['all_q0_q1_q2_correct_num']}/{summary['all_q0_q1_q2_correct_den']} = {fmt_rate(summary['all_q0_q1_q2_correct_rate'])}")
        log(f"Parsed ans:   {summary['parsed_answer_num']}/{summary['parsed_answer_den']} = {fmt_rate(summary['parsed_answer_rate'])}")
        log(f"No JSON out:  {summary['no_json_output_num']}/{summary['no_json_output_den']} = {fmt_rate(summary['no_json_output_rate'])}")

        print("=" * 80)
        print(f"base model:   {runner.base_model_path}")
        print(f"adapter:      {runner.adapter_path}")
        print("method:       grouped_no_json_q0_q1_q2")
        print(f"image_mode:   {args.image_mode}")
        print(f"test_dir:     {test_dir}")
        print(f"groups:       {summary['num_groups']}")
        print(f"items:        {summary['num_items']}")
        print(f"accuracy:     {summary['accuracy_num']}/{summary['accuracy_den']} = {fmt_rate(summary['accuracy'])}")
        print(f"all-correct:  {summary['all_q0_q1_q2_correct_num']}/{summary['all_q0_q1_q2_correct_den']} = {fmt_rate(summary['all_q0_q1_q2_correct_rate'])}")
        print(f"score:        {summary['score_num']}/{summary['score_den']} = {fmt_rate(summary['score'])}")
        print(f"parsed ans:   {summary['parsed_answer_num']}/{summary['parsed_answer_den']} = {fmt_rate(summary['parsed_answer_rate'])}")
        print(f"no JSON out:  {summary['no_json_output_num']}/{summary['no_json_output_den']} = {fmt_rate(summary['no_json_output_rate'])}")

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
