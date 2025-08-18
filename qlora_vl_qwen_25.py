import os, io, json, argparse, random, base64, re, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import pandas as pd
import numpy as np
from PIL import Image
from urllib.request import Request, urlopen
from urllib.parse import urlparse

from sklearn.model_selection import StratifiedShuffleSplit

from transformers import (
    AutoModelForCausalLM, AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==========================
# Config
# ==========================
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUT_ROOT = "/content/drive/MyDrive/Colab Notebooks/wook/output"
DATA_SUBDIR = "datasets/qlora"

SYSTEM_PROMPT = (
    "You are a single vision–language assistant. You will receive either an image or text, and optionally a question.\n"
    "Always reply in English in a single, unified style across tasks.\n"
    "If the answer cannot be determined from the provided input, output exactly: unknown."
)

LONG_SIDE = 1280
MAX_SIDE_HARD = 3500
URL_TIMEOUT = 20
IMG_BASE: Optional[str] = None  # can be set by caller
MIN_IMG_SIDE = 64  # reject images smaller than this

RESIZE_CACHE_DIR = "/tmp/img_resized_cache"
os.makedirs(RESIZE_CACHE_DIR, exist_ok=True)

# ==========================
# Utils
# ==========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl") or path.endswith(".json"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def to_unified_schema(df: pd.DataFrame, col_map: Optional[dict] = None) -> pd.DataFrame:
    df = df.copy()
    if col_map:
        df = df.rename(columns=col_map)
    required_base = ["task","input_type","input"]
    for c in required_base:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if "question" not in df.columns:
        df["question"] = ""
    if "output" not in df.columns:
        df["output"] = ""
    for c in ["task","input_type","question","output"]:
        df[c] = df[c].astype(str)
    def _norm_it(x: str) -> str:
        x = str(x).lower().strip()
        if "img" in x or x == "image":
            return "image"
        return "text" if "text" in x or x == "text" else x
    df["input_type"] = df["input_type"].map(_norm_it)
    return df[["task","input_type","input","question","output"]]


def join_labels_generic(inputs_df: pd.DataFrame, labels_df: pd.DataFrame,
                        id_col: str = "id", label_col: str = "output") -> pd.DataFrame:
    inputs = inputs_df.copy()
    labels = labels_df.copy()
    if id_col in inputs.columns and id_col in labels.columns:
        labels = labels[[id_col, label_col]].copy()
        labels[id_col] = labels[id_col].astype(str)
        inputs[id_col] = inputs[id_col].astype(str)
        out = inputs.merge(labels, on=id_col, how="left", suffixes=("","_lab"))
        cond = (out["output"].astype(str).str.len() == 0) | out["output"].isna()
        out.loc[cond, "output"] = out.loc[cond, f"{label_col}_lab"].astype(str)
        out = out.drop(columns=[f"{label_col}_lab"]) if f"{label_col}_lab" in out.columns else out
        return out
    if len(inputs) == len(labels) and label_col in labels.columns:
        out = inputs.copy()
        out["output"] = labels[label_col].astype(str).values
        return out
    raise ValueError("Could not align labels: need shared id column or equal lengths with label_col present")


def stratified_split(df: pd.DataFrame, valid_ratio: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key = df["task"].astype(str) + "||" + df["input_type"].astype(str)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=seed)
    idx = np.arange(len(df))
    train_idx, valid_idx = next(splitter.split(idx, key))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(drop=True)


def save_both_parquet_jsonl(df: pd.DataFrame, path_wo_ext: str):
    os.makedirs(os.path.dirname(path_wo_ext), exist_ok=True)
    df.to_parquet(path_wo_ext + ".parquet", index=False)
    with open(path_wo_ext + ".jsonl", "w", encoding="utf-8") as f:
        for r in df.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_train_valid(out_root: str,
                      inputs_file: str,
                      labels_file: Optional[str] = None,
                      valid_ratio: float = 0.1,
                      col_map_inputs: Optional[dict] = None,
                      col_map_labels: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = os.path.join(out_root, DATA_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    inputs = read_any(inputs_file)
    inputs = to_unified_schema(inputs, col_map_inputs)
    if labels_file:
        labels = read_any(labels_file)
        if col_map_labels:
            labels = labels.rename(columns=col_map_labels)
        data = join_labels_generic(inputs, labels)
    else:
        data = inputs
        data = data[data["output"].astype(str).str.len() > 0].reset_index(drop=True)
    train_df, valid_df = stratified_split(data, valid_ratio=valid_ratio, seed=42)
    save_both_parquet_jsonl(train_df, os.path.join(out_dir, "train"))
    save_both_parquet_jsonl(valid_df, os.path.join(out_dir, "valid"))
    return train_df, valid_df

# ==========================
# Image I/O (strict)
# ==========================
_b64_re = re.compile(r'^[A-Za-z0-9+/=\n\r]+$')

def looks_like_base64(s: str, min_len: int = 128) -> bool:
    if not (isinstance(s, str) and len(s) >= min_len and _b64_re.match(s)):
        return False
    try:
        head = base64.b64decode(s[:4096], validate=True)
        return head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8")
    except Exception:
        return False

def is_url(path: str) -> bool:
    try:
        return urlparse(str(path)).scheme in ("http", "https")
    except Exception:
        return False


def _cap_max_side(img: Image.Image, cap=MAX_SIDE_HARD) -> Image.Image:
    if max(img.size) <= cap:
        return img
    img = img.copy()
    img.thumbnail((cap, cap))
    return img


def _resize_keep_ratio(img: Image.Image, long_side: int) -> Image.Image:
    if max(img.size) <= long_side:
        return img
    img = img.copy()
    img.thumbnail((long_side, long_side))
    return img


def _hash_image_bytes(img: Image.Image) -> str:
    with io.BytesIO() as bio:
        img.save(bio, format="PNG", optimize=False)
        return hashlib.md5(bio.getvalue()).hexdigest()


def finalize_image(img: Image.Image) -> Image.Image:
    img = _cap_max_side(img, MAX_SIDE_HARD)
    target = LONG_SIDE
    if max(img.size) <= target:
        return img
    h = _hash_image_bytes(img) + f"_{target}"
    path = os.path.join(RESIZE_CACHE_DIR, h + ".png")
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    out = _resize_keep_ratio(img, target)
    out.save(path, format="PNG")
    return out


def load_image_strict(input_obj) -> Image.Image:
    """Load an image, raise on failure. No 1×1 fallbacks."""
    try:
        if isinstance(input_obj, (bytes, bytearray)):
            img = Image.open(io.BytesIO(input_obj)).convert("RGB")
            img = finalize_image(img)
        elif isinstance(input_obj, str):
            s = input_obj.strip()
            if s.startswith("data:image"):
                b64 = s.split(",", 1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                img = finalize_image(img)
            elif looks_like_base64(s):
                img = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
                img = finalize_image(img)
            elif is_url(s):
                req = Request(s, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=URL_TIMEOUT) as r:
                    raw = r.read()
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                img = finalize_image(img)
            else:
                p = s
                if not os.path.isabs(p) and IMG_BASE:
                    p = os.path.join(IMG_BASE, p)
                img = Image.open(p).convert("RGB")
                img = finalize_image(img)
        else:
            raise TypeError(f"Unsupported image input type: {type(input_obj)}")
    except Exception as e:
        raise RuntimeError(f"[ImageLoadError] {input_obj!r} -> {e}")

    w, h = img.size
    if min(w, h) < MIN_IMG_SIDE:
        raise ValueError(f"[ImageTooSmall] {w}x{h} < {MIN_IMG_SIDE}")
    return img

# ==========================
# Prompt building (aligned with inference style)
# ==========================

def build_user_prompt(task: str, input_type: str, the_input: str, question: str | None) -> str:
    t = (task or "").strip().lower()
    it = (input_type or "text").strip().lower()
    q = (question or "").strip()
    lines = [
        f"Task: {t}",
        f"InputType: {it}",
        f"Question: {q}" if q else "Question:",
    ]
    lines.append("Input:\n" + (the_input or "") if it == "text" else "Input: <image>")
    return "\n".join(lines)


def build_messages_placeholder(inp_type: str, task: str, inp: str, question: str, output: Optional[str] = None,
                               add_task_hint: bool = False) -> Tuple[List[Dict[str, Any]], Optional[List[Image.Image]]]:
    # Build messages with <image> placeholder; pass PIL images separately
    hint = f"[TASK={task}]\n" if add_task_hint else ""
    user_prompt = build_user_prompt(task, inp_type, str(inp), question)

    if inp_type == "image":
        # Load image strictly and keep only placeholder in content
        img = load_image_strict(inp)
        img = finalize_image(img)
        user_content = [{"type": "image"}, {"type": "text", "text": hint + user_prompt}]
        images = [img]
    else:
        user_content = [{"type": "text", "text": hint + user_prompt}]
        images = None

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": user_content},
    ]
    if output is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": str(output).rstrip()}]})
    return msgs, images

# ==========================
# Dataset & Collator
# ==========================

class MMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: AutoProcessor, add_task_hint: bool = False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.add_task_hint = add_task_hint
        self.max_length = getattr(self.processor.tokenizer, 'model_max_length', 4096)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx].to_dict()
        inp_type = str(row.get("input_type")).strip().lower()
        task     = str(row.get("task")).strip()
        inp      = row.get("input")
        q        = row.get("question") or ""
        out      = str(row.get("output")).rstrip()

        msgs_prompt_only, images = build_messages_placeholder(
            inp_type, task, inp, q, output=None, add_task_hint=self.add_task_hint
        )
        msgs_with_answer, _ = build_messages_placeholder(
            inp_type, task, inp, q, output=out, add_task_hint=self.add_task_hint
        )

        prompt_text = self.processor.apply_chat_template(
            msgs_prompt_only, tokenize=False, add_generation_prompt=False
        )
        full_text = self.processor.apply_chat_template(
            msgs_with_answer, tokenize=False, add_generation_prompt=False
        )

        # Encode with same images for prompt/full
        if images is not None:
            enc_full = self.processor(
                text=full_text, images=images,
                return_tensors="pt", padding="longest",
                # ✦ truncation 금지: max_length도 주지 않음
            )
            enc_prompt = self.processor(
                text=prompt_text, images=images,
                return_tensors="pt", padding="longest",
            )
        else:
            enc_full = self.processor(
                text=full_text,
                return_tensors="pt", padding="longest",
                truncation=True, max_length=self.max_length,  # 텍스트만 안전
            )
            enc_prompt = self.processor(
                text=prompt_text,
                return_tensors="pt", padding="longest",
                truncation=True, max_length=self.max_length,
            )

        input_ids = enc_full["input_ids"][0]
        attn_mask = enc_full["attention_mask"][0]
        labels = input_ids.clone()
        prompt_len = enc_prompt["input_ids"].shape[-1]
        labels[:prompt_len] = -100
        
        item = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }
        for k, v in enc_full.items():
            if k in item:
                continue
            # ★ 텍스트 3형제 외에는 절대 [0] 하지 말 것!
            item[k] = v
        return item


@dataclass
class QwenVLDataCollator:
    processor  # AutoProcessor
    pad_token_id: int

    def __call__(self, features):
        # 1) 텍스트 라벨은 이미 각 item에 있으므로 pad에서 함께 패딩
        #    (pad 후 라벨의 pad 토큰을 -100으로 바꾸는 보정만 해주면 더 안전)
        batch = self.processor.pad(features, return_tensors="pt")

        # 2) labels 패딩(-100) 보정
        if "labels" in batch:
            labels = batch["labels"]
            labels = labels.masked_fill(labels == self.pad_token_id, -100)
            batch["labels"] = labels

        return batch

# ==========================
# QLoRA Config
# ==========================

def make_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # A100 supports bf16
    )


def make_lora_config(r=64, alpha=128, dropout=0.05, target_modules: Optional[List[str]] = None) -> LoraConfig:
    if target_modules is None:
        target_modules = [
            "q_proj","k_proj","v_proj","o_proj",
            "up_proj","down_proj","gate_proj",
        ]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def resolve_profile(profile: str) -> dict:
    profile = (profile or "balanced").lower()
    if profile == "conservative":
        return dict(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=1e-4,
            num_train_epochs=1,
            warmup_ratio=0.03,
            weight_decay=0.0,
            logging_steps=25,
            save_steps=1000,
            eval_steps=500,
        )
    if profile == "aggressive":
        return dict(
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            num_train_epochs=1,
            warmup_ratio=0.03,
            weight_decay=0.01,
            logging_steps=20,
            save_steps=1000,
            eval_steps=400,
        )
    return dict(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=1,
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=25,
        save_steps=1000,
        eval_steps=500,
    )

# ==========================
# Train
# ==========================

def train_model(
    base_model: str,
    train_file: str,
    valid_file: Optional[str],
    out_root: str,
    profile: str = "balanced",
    add_task_hint: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    set_seed(42)
    run_dir = os.path.join(out_root, f"qlora-vl-qwen25-7b")
    os.makedirs(run_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        min_pixels=256*28*28,
        max_pixels=1024*28*28,
    )

    tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = make_bnb_config()
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    except Exception as _e1:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        except Exception as _e2:
            raise RuntimeError(f"Failed to load VLM base model for {base_model}: {_e1} / {_e2}")

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_cfg = make_lora_config(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target_modules=target_modules)
    model = get_peft_model(model, peft_cfg)

    train_df = read_any(train_file)
    valid_df = read_any(valid_file) if valid_file else None

    required = {"task","input_type","input","question","output"}
    if not required.issubset(set(train_df.columns)):
        missing = required - set(train_df.columns)
        raise ValueError(f"Train file missing columns: {missing}")
    if valid_df is not None and not required.issubset(set(valid_df.columns)):
        missing = required - set(valid_df.columns)
        raise ValueError(f"Valid file missing columns: {missing}")

    # quick audit for image rows (fail fast)
    if (train_df["input_type"].str.lower() == "image").any():
        for i, r in train_df[(train_df["input_type"].str.lower()=="image")].head(8).iterrows():  # sample a few
            _ = load_image_strict(r["input"])  # will raise on bad/too small

    train_ds = MMDataset(train_df, processor, add_task_hint=add_task_hint)
    eval_ds = MMDataset(valid_df, processor, add_task_hint=False) if valid_df is not None else None
    collator = VLDataCollator(pad_token_id=tokenizer.pad_token_id)

    p = resolve_profile(profile)
    import inspect
    ta_kwargs = dict(
        output_dir=run_dir,
        per_device_train_batch_size=p["per_device_train_batch_size"],
        per_device_eval_batch_size=p["per_device_eval_batch_size"],
        gradient_accumulation_steps=p["gradient_accumulation_steps"],
        learning_rate=p["learning_rate"],
        num_train_epochs=p["num_train_epochs"],
        warmup_ratio=p["warmup_ratio"],
        weight_decay=p["weight_decay"],
        logging_steps=p["logging_steps"],
        save_steps=p["save_steps"],
        save_total_limit=2,
        bf16=True,  # A100
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,  # keep vision fields
        optim="paged_adamw_8bit",
    )

    sig = inspect.signature(TrainingArguments.__init__)
    def maybe(key, value):
        if key in sig.parameters and value is not None:
            ta_kwargs[key] = value

    eval_strategy = "steps" if eval_ds is not None else "no"
    maybe("evaluation_strategy", eval_strategy)
    maybe("eval_strategy", eval_strategy)
    maybe("eval_steps", p.get("eval_steps") if eval_ds is not None else None)
    maybe("load_best_model_at_end", bool(eval_ds))
    maybe("metric_for_best_model", "eval_loss" if eval_ds is not None else None)
    maybe("greater_is_better", False if eval_ds is not None else None)

    training_args = TrainingArguments(**ta_kwargs)

    init_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        init_kwargs["processing_class"] = processor
    else:
        init_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**init_kwargs)
    trainer.train()

    trainer.model.save_pretrained(run_dir)
    processor.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    return run_dir

# ==========================
# Quick infer (optional, to validate the adapter after training)
# ==========================

def quick_infer(run_dir: str, base_model: str = DEFAULT_BASE_MODEL, image_url: Optional[str] = None, question: Optional[str] = None):
    from peft import PeftModel
    try:
        model = AutoModelForVision2Seq.from_pretrained(base_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

    model = PeftModel.from_pretrained(model, run_dir)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        run_dir,    # adapter 저장 경로
        trust_remote_code=True,
        min_pixels=256*28*28,
        max_pixels=1024*28*28,
    )
    tok = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)

    msgs = [{"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT}]}]
    images = None
    if image_url:
        req = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=URL_TIMEOUT) as r:
            raw = r.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = finalize_image(img)
        msgs.append({"role":"user","content":[{"type":"image"},{"type":"text","text":question or "What is in the image?"}]})
        images = [img]
    else:
        msgs.append({"role":"user","content":[{"type":"text","text":question or "Please provide a short answer."}]})

    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    ids = out[0]
    try:
        text = tok.decode(ids, skip_special_tokens=True)
    except Exception:
        text = processor.decode(ids, skip_special_tokens=True)
    return text

# ==========================
# CLI
# ==========================

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p_build.add_argument("--inputs_file", type=str, required=True)
    p_build.add_argument("--labels_file", type=str, default=None)
    p_build.add_argument("--valid_ratio", type=float, default=0.1)

    p_train = sub.add_parser("train")
    p_train.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    p_train.add_argument("--train_file", type=str, required=True)
    p_train.add_argument("--valid_file", type=str, default=None)
    p_train.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p_train.add_argument("--profile", type=str, default="balanced", choices=["conservative","balanced","aggressive"])
    p_train.add_argument("--add_task_hint", action="store_true")
    p_train.add_argument("--lora_r", type=int, default=64)
    p_train.add_argument("--lora_alpha", type=int, default=128)
    p_train.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    if args.cmd == "build":
        build_train_valid(
            out_root=args.out_root,
            inputs_file=args.inputs_file,
            labels_file=args.labels_file,
            valid_ratio=args.valid_ratio,
        )
        return

    if args.cmd == "train":
        train_model(
            base_model=args.base_model,
            train_file=args.train_file,
            valid_file=args.valid_file,
            out_root=args.out_root,
            profile=args.profile,
            add_task_hint=args.add_task_hint,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        return


if __name__ == "__main__":
    main()
