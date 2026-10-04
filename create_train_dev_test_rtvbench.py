from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import Video, load_dataset
from huggingface_hub import hf_hub_download


QA_JSON_CANDIDATES = [
    ("dataset", "RTVBench/RTV-Bench", "QA.json"),
    ("model", "LJungang/RTV-Bench", "QA.json"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split RTV-Bench into train/dev/test folders."
    )
    p.add_argument("--dataset_repo", default="RTVBench/RTV-Bench")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--qa_json", type=str, default=None)
    p.add_argument("--output_root", type=str, required=True)

    # Safer default: split by video to avoid leakage.
    p.add_argument(
        "--split_unit",
        choices=["video", "group", "question"],
        default="video",
        help="Unit used to assign data to a split.",
    )

    p.add_argument("--train_ratio", type=float, default=0.60)
    p.add_argument("--dev_ratio", type=float, default=0.20)
    p.add_argument("--test_ratio", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--video_mode",
        choices=["copy", "symlink", "hardlink", "none"],
        default="symlink",
        help="How to place videos into split folders.",
    )

    p.add_argument(
        "--video_subdir",
        default="videos",
        help="Subdirectory name inside each split folder for videos.",
    )

    return p.parse_args()


def maybe_download_qa_json(explicit_path: str | None) -> str:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"--qa_json not found: {path}")
        return str(path)

    for repo_type, repo_id, filename in QA_JSON_CANDIDATES:
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
            )
        except Exception:
            pass

    raise RuntimeError(
        "Could not automatically find QA.json. Pass --qa_json explicitly."
    )


def load_qa_annotations(qa_json_path: str) -> List[Dict[str, Any]]:
    with open(qa_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("QA.json must contain a list of annotation dicts.")

    return data


def load_video_dataset(repo_id: str, split: str):
    ds = load_dataset(repo_id, split=split)
    ds = ds.cast_column("video", Video(decode=False))
    return ds


def build_video_index(ds) -> Dict[str, str]:
    """
    Maps video basename -> full local path in Hugging Face cache.
    """
    out: Dict[str, str] = {}
    for row in ds:
        video_info = row.get("video")
        if not isinstance(video_info, dict):
            continue
        path = video_info.get("path")
        if not path:
            continue
        out[os.path.basename(path)] = path
    return out


def parse_group_id(question_id: Any) -> str | None:
    """
    Example expected format:
    q-group-<groupid>-0
    q-group-<groupid>-1
    q-group-<groupid>-2
    possibly with extra suffixes
    """
    if not isinstance(question_id, str):
        return None

    parts = question_id.strip().split("-")
    if len(parts) < 4:
        return None
    if parts[0] != "q" or parts[1] != "group":
        return None

    return parts[2]


def get_split_key(item: Dict[str, Any], split_unit: str) -> str:
    if split_unit == "video":
        video_name = item.get("video")
        if not isinstance(video_name, str):
            raise ValueError(f"Missing/invalid video field in item: {item}")
        return f"video::{video_name}"

    if split_unit == "group":
        gid = parse_group_id(item.get("questionID"))
        if gid is None:
            raise ValueError(f"Could not parse group id from questionID: {item.get('questionID')}")
        return f"group::{gid}"

    if split_unit == "question":
        qid = item.get("questionID")
        if not isinstance(qid, str):
            raise ValueError(f"Missing/invalid questionID in item: {item}")
        return f"question::{qid}"

    raise ValueError(f"Unknown split_unit: {split_unit}")


def group_items_by_unit(
    annotations: List[Dict[str, Any]],
    split_unit: str,
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        key = get_split_key(item, split_unit)
        groups[key].append(item)
    return groups


def assign_keys_to_splits(
    keys: List[str],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    total = train_ratio + dev_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio} + {dev_ratio} + {test_ratio} = {total}"
        )

    rng = random.Random(seed)
    shuffled = list(keys)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_dev = int(round(n * dev_ratio))

    # Ensure all keys are assigned, with test getting the remainder.
    if n_train > n:
        n_train = n
    if n_train + n_dev > n:
        n_dev = n - n_train

    train_keys = set(shuffled[:n_train])
    dev_keys = set(shuffled[n_train:n_train + n_dev])
    test_keys = set(shuffled[n_train + n_dev:])

    mapping: Dict[str, str] = {}
    for k in train_keys:
        mapping[k] = "train"
    for k in dev_keys:
        mapping[k] = "dev"
    for k in test_keys:
        mapping[k] = "test"

    return mapping


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def place_video(src: Path, dst: Path, mode: str) -> None:
    if mode == "none":
        return

    ensure_dir(dst.parent)
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return

    if mode == "hardlink":
        os.link(src, dst)
        return

    raise ValueError(f"Unknown video mode: {mode}")


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    qa_json_path = maybe_download_qa_json(args.qa_json)
    annotations = load_qa_annotations(qa_json_path)

    ds = load_video_dataset(args.dataset_repo, args.dataset_split)
    video_index = build_video_index(ds)

    grouped = group_items_by_unit(annotations, args.split_unit)
    split_map = assign_keys_to_splits(
        keys=sorted(grouped.keys()),
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_items: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "test": [],
    }

    split_videos: Dict[str, set[str]] = {
        "train": set(),
        "dev": set(),
        "test": set(),
    }

    for key, items in grouped.items():
        split_name = split_map[key]
        split_items[split_name].extend(items)
        for item in items:
            video_name = item.get("video")
            if isinstance(video_name, str):
                split_videos[split_name].add(video_name)

    output_root = Path(args.output_root)
    for split_name in ("train", "dev", "test"):
        split_dir = output_root / split_name
        videos_dir = split_dir / args.video_subdir

        ensure_dir(split_dir)
        if args.video_mode != "none":
            ensure_dir(videos_dir)

        # Write QA.json for the split
        split_qa = sorted(
            split_items[split_name],
            key=lambda x: (
                str(x.get("video", "")),
                float(x.get("end_time", 0.0)),
                str(x.get("questionID", "")),
            ),
        )
        write_json(split_dir / "QA.json", split_qa)

        # Place videos for the split
        missing_videos: List[str] = []
        for video_name in sorted(split_videos[split_name]):
            src_path = video_index.get(video_name)
            if src_path is None:
                missing_videos.append(video_name)
                continue

            src = Path(src_path)
            dst = videos_dir / video_name
            place_video(src, dst, args.video_mode)

        # Write metadata summary
        summary = {
            "split_name": split_name,
            "num_items": len(split_items[split_name]),
            "num_videos": len(split_videos[split_name]),
            "split_unit": args.split_unit,
            "video_mode": args.video_mode,
            "missing_videos": missing_videos,
        }
        write_json(split_dir / "split_info.json", summary)

    # Top-level summary
    global_summary = {
        "total_items": len(annotations),
        "train_items": len(split_items["train"]),
        "dev_items": len(split_items["dev"]),
        "test_items": len(split_items["test"]),
        "train_videos": len(split_videos["train"]),
        "dev_videos": len(split_videos["dev"]),
        "test_videos": len(split_videos["test"]),
        "split_unit": args.split_unit,
        "ratios": {
            "train": args.train_ratio,
            "dev": args.dev_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
        "video_mode": args.video_mode,
    }
    write_json(output_root / "split_summary.json", global_summary)

    print("Done.")
    print(json.dumps(global_summary, indent=2))


if __name__ == "__main__":
    main()