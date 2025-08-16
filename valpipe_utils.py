"""
valpipe_utils.py  (v2)
- Step 3: Stratified data split (train/dev) + optional dev cap per task
- Step 4: Task-wise metrics + overall equal-weight score
- NEW: Balanced subsampling utilities to shorten inference time (keep input_type ratio)

Assumed columns in your dataset:
    id, task, input_type, input, question, output
"""

from __future__ import annotations
import os, re, math, csv, json, random
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

__all__ = [
    "safe_to_csv_utf8", "assert_utf8_ok",
    "stratified_split_by_task", "save_splits",
    "sample_per_task", "cap_per_task",
    "normalize_answer", "extract_math_final_number", "norm_num",
    "metric_caption_clipscore", "metric_accuracy", "metric_math_em", "metric_bertscore",
    "evaluate_from_df", "evaluate_from_file", "MetricsResult"
]

# ---------------------------
# Encoding helpers
# ---------------------------
def safe_to_csv_utf8(df: pd.DataFrame, path: str):
    """Save as UTF-8 CSV with safe quoting and no index; pandas version-agnostic line terminator."""
    try:
        df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    except TypeError:
        df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

def assert_utf8_ok(path: str):
    """Quick sanity check: UTF-8 decodable and csv.reader parsable (first few rows)."""
    raw = Path(path).read_bytes()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise AssertionError(f"Not UTF-8: {e}")
    import csv as _csv
    with open(path, "r", encoding="utf-8", newline="") as f:
        _ = list(_csv.reader(f))[:5]

# ---------------------------
# Normalization helpers
# ---------------------------
_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^a-z0-9 ]+")

def normalize_answer(s: str) -> str:
    """Lower, strip punctuation, collapse spaces."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = _ws_re.sub(" ", s)
    s = _punct_re.sub("", s)
    s = _ws_re.sub(" ", s).strip()
    return s

_num_re = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")
def extract_math_final_number(s: str) -> str:
    """Extract final number after '#### ' if present; else last number in string."""
    if not isinstance(s, str):
        return ""
    m = _num_re.search(s)
    if m:
        return m.group(1)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else ""

def norm_num(x: str) -> str:
    """Normalize numeric string for exact-match comparison."""
    if x is None or x == "":
        return ""
    try:
        v = float(x)
        return str(int(v)) if v.is_integer() else str(v).rstrip("0").rstrip(".")
    except Exception:
        return str(x).strip()

# ---------------------------
# Internal: proportional allocation helper
# ---------------------------
def _alloc_counts(group_sizes: List[int], total: int) -> List[int]:
    """
    Largest Remainder Method: proportionally allocate 'total' across buckets of sizes group_sizes.
    Ensures sum == total and each bucket <= original size.
    """
    if total <= 0 or sum(group_sizes) == 0:
        return [0] * len(group_sizes)
    props = np.array(group_sizes, dtype=float) / float(sum(group_sizes))
    raw = props * total
    base = np.floor(raw).astype(int).tolist()
    rem = total - sum(base)
    # distribute leftovers by largest fractional parts
    fracs = [(i, raw[i] - base[i]) for i in range(len(group_sizes))]
    fracs.sort(key=lambda x: x[1], reverse=True)
    for i, _ in fracs:
        if rem <= 0:
            break
        take = min(1, group_sizes[i] - base[i])
        if take > 0:
            base[i] += take
            rem -= take
    return base

# ---------------------------
# Step 3: Stratified split (+ optional dev cap)
# ---------------------------
def stratified_split_by_task(df: pd.DataFrame, dev_ratio: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-task stratified split. Returns (train_df, dev_df)."""
    rng = np.random.default_rng(seed)
    parts = []
    for t, g in df.groupby(df["task"].str.lower()):
        idx = np.arange(len(g))
        rng.shuffle(idx)
        cut = int(len(g) * (1 - dev_ratio))
        train_idx = g.index.values[idx[:cut]]
        dev_idx   = g.index.values[idx[cut:]]
        parts.append(("train", df.loc[train_idx]))
        parts.append(("dev",   df.loc[dev_idx]))
    train = pd.concat([p[1] for p in parts if p[0]=="train"]).sort_index().reset_index(drop=True)
    dev   = pd.concat([p[1] for p in parts if p[0]=="dev"]).sort_index().reset_index(drop=True)
    return train, dev

def sample_per_task(df: pd.DataFrame, per_task: int, seed: int = 42, keep_input_type_ratio: bool = True) -> pd.DataFrame:
    """
    Balanced subsample: pick up to 'per_task' rows from each task.
    If keep_input_type_ratio=True, preserve input_type proportion within each task.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for t, g in df.groupby(df["task"].str.lower(), sort=False):
        n_take = min(len(g), int(per_task))
        if n_take <= 0:
            continue
        if keep_input_type_ratio and "input_type" in g.columns:
            # allocate per input_type
            counts = g["input_type"].str.lower().value_counts()
            keys = counts.index.tolist()
            allocs = _alloc_counts([counts[k] for k in keys], n_take)
            for k, a in zip(keys, allocs):
                if a > 0:
                    sub = g[g["input_type"].str.lower() == k].sample(n=a, random_state=int(rng.integers(0, 1<<32)))
                    chunks.append(sub)
        else:
            chunks.append(g.sample(n=n_take, random_state=int(rng.integers(0, 1<<32))))
    if not chunks:
        return df.iloc[0:0].copy()
    out = pd.concat(chunks).reset_index(drop=True)
    return out

def cap_per_task(df: pd.DataFrame, max_per_task: int, seed: int = 42, keep_input_type_ratio: bool = True) -> pd.DataFrame:
    """Alias for sample_per_task with clearer intent."""
    return sample_per_task(df, per_task=max_per_task, seed=seed, keep_input_type_ratio=keep_input_type_ratio)

def save_splits(df: pd.DataFrame, out_dir: str, dev_ratio: float = 0.2, seed: int = 42,
                dev_max_per_task: Optional[int] = None, keep_input_type_ratio: bool = True) -> tuple[str, str]:
    """
    Split and save to parquet files. Optionally cap dev to N per task (keeping input_type ratio).
    Returns (train_path, dev_path).
    """
    os.makedirs(out_dir, exist_ok=True)
    train_df, dev_df = stratified_split_by_task(df, dev_ratio=dev_ratio, seed=seed)
    if dev_max_per_task is not None:
        dev_df = cap_per_task(dev_df, dev_max_per_task, seed=seed, keep_input_type_ratio=keep_input_type_ratio)
    train_path = os.path.join(out_dir, "train.parquet")
    dev_path   = os.path.join(out_dir, "dev.parquet")
    train_df.to_parquet(train_path, index=False)
    dev_df.to_parquet(dev_path, index=False)
    return train_path, dev_path

# ---------------------------
# Step 4: Metrics
# ---------------------------
@dataclass
class MetricsResult:
    by_task: Dict[str, float]
    overall_equal_weight: float

_to_tensor = transforms.Compose([transforms.ToTensor()])

def _default_image_loader(path_or_url: str) -> Image.Image:
    """Minimal loader: local path only. Override with a robust loader if needed."""
    try:
        return Image.open(path_or_url).convert("RGB")
    except Exception:
        return Image.new("RGB", (1,1), (0,0,0))

def metric_caption_clipscore(df_cap: pd.DataFrame, image_loader: Optional[Callable[[str], Image.Image]] = None, device: Optional[str] = None) -> float:
    """Compute CLIPScore for captioning samples comparing predicted captions to input images."""
    if len(df_cap) == 0:
        return float("nan")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from torchmetrics.multimodal import CLIPScore
    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    loader = image_loader or _default_image_loader
    scores = []
    B = 16
    for s in range(0, len(df_cap), B):
        chunk = df_cap.iloc[s:s+B]
        imgs = [_to_tensor(loader(str(x))).to(device) for x in chunk["input"].tolist()]
        texts = chunk["pred"].tolist()
        with torch.no_grad():
            sc = clip(imgs, texts)  # mean over batch
        scores.extend([float(sc)] * len(chunk))
    return float(np.mean(scores))

def metric_accuracy(df_task: pd.DataFrame) -> float:
    """Exact-match accuracy with normalized strings (for VQA, text_qa)."""
    if len(df_task) == 0:
        return float("nan")
    gold = df_task["output"].map(normalize_answer).tolist()
    pred = df_task["pred"].map(normalize_answer).tolist()
    return float(np.mean([p == g for p, g in zip(pred, gold)]))

def metric_math_em(df_math: pd.DataFrame) -> float:
    """Exact match of final numbers for math reasoning."""
    if len(df_math) == 0:
        return float("nan")
    gold = df_math["output"].map(extract_math_final_number).map(norm_num).tolist()
    pred = df_math["pred"].map(extract_math_final_number).map(norm_num).tolist()
    return float(np.mean([p == g for p, g in zip(pred, gold)]))

def metric_bertscore(df_sum: pd.DataFrame) -> float:
    """BERTScore F1 for summarization (evaluate)."""
    if len(df_sum) == 0:
        return float("nan")
    import evaluate
    bertscore = evaluate.load("bertscore")
    res = bertscore.compute(
        predictions=df_sum["pred"].tolist(),
        references=df_sum["output"].tolist(),
        lang="en"
    )
    return float(np.mean(res.get("f1", []))) if res else float("nan")

def evaluate_from_df(df_pred: pd.DataFrame, image_loader: Optional[Callable[[str], Image.Image]] = None) -> MetricsResult:
    """Compute all task metrics and equal-weight mean from a predictions dataframe
    with columns: id, task, input_type, input, question, output, pred
    """
    t = df_pred["task"].str.lower()
    m_cap  = t == "captioning"
    m_vqa  = t == "vqa"
    m_math = t == "math_reasoning"
    m_icqa = t == "text_qa"
    m_sum  = t == "summarization"

    by_task = {}
    by_task["captioning"]     = metric_caption_clipscore(df_pred[m_cap], image_loader=image_loader)
    by_task["vqa"]            = metric_accuracy(df_pred[m_vqa])
    by_task["math_reasoning"] = metric_math_em(df_pred[m_math])
    by_task["text_qa"]        = metric_accuracy(df_pred[m_icqa])
    by_task["summarization"]  = metric_bertscore(df_pred[m_sum])

    vals = [by_task[k] for k in ["captioning","vqa","math_reasoning","text_qa","summarization"]]
    overall = float(np.mean(vals)) if all(not (isinstance(v, float) and math.isnan(v)) for v in vals) else float("nan")

    return MetricsResult(by_task=by_task, overall_equal_weight=overall)

def evaluate_from_file(pred_csv: str, image_loader: Optional[Callable[[str], Image.Image]] = None) -> MetricsResult:
    """Load predictions CSV (UTF-8) and evaluate. CSV must contain pred column."""
    df = pd.read_csv(pred_csv, dtype=str, encoding="utf-8")
    return evaluate_from_df(df, image_loader=image_loader)
