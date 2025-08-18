"""
QLoRA (4-bit) fine-tuning for a Vision–Language model (Qwen2.5-VL-7B-Instruct)
- Single adapter (LoRA) on LANGUAGE MODULES ONLY (phase 1)
- Single unified prompt across tasks
- Label masking (only assistant tokens contribute to loss)
- Dataset builder (train/valid) included

USAGE (Colab/CLI):

# 0) (Optional) install
# !pip install -U "transformers>=4.46" "accelerate>=0.34" "peft>=0.11" bitsandbytes pandas pillow requests scikit-learn

# 1) Build train/valid from inputs(+labels)
# python qlora_vl_qwen25_complete.py build \
#   --out_root "/content/drive/MyDrive/Colab Notebooks/wook/output" \
#   --inputs_file /path/to/inputs.parquet \
#   --labels_file /path/to/mygold.parquet \
#   --valid_ratio 0.1

# 2) Train (Balanced profile by default)
# python qlora_vl_qwen25_complete.py train \
#   --base_model Qwen/Qwen2.5-VL-7B-Instruct \
#   --train_file "/content/drive/MyDrive/Colab Notebooks/wook/output/datasets/qlora/train.parquet" \
#   --valid_file "/content/drive/MyDrive/Colab Notebooks/wook/output/datasets/qlora/valid.parquet" \
#   --out_root "/content/drive/MyDrive/Colab Notebooks/wook/output" \
#   --profile balanced \
#   --add_task_hint

# 3) (Optional) Quick inference after training: see `quick_infer()` at bottom.

"""

import os, io, json, math, argparse, random, time, sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import pandas as pd
import numpy as np
from PIL import Image
import requests
import base64

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

# =============================
# Globals & Defaults
# =============================
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUT_ROOT = "/content/drive/MyDrive/Colab Notebooks/wook/output"
DATA_SUBDIR = "datasets/qlora"

SYSTEM_PROMPT = (
    "You are a single vision–language assistant. You will receive either an image or text, and optionally a question.\n"
    "Always reply in English in a single, unified style across tasks.\n"
    "If the answer cannot be determined from the provided input, output exactly: unknown."
)

# =============================
# Utils
# =============================

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
    """
    Standardize to columns: [task, input_type, input, question, output]
    input_type ∈ {"image", "text"}; input can be path/URL, raw bytes/base64, or text.
    NOTE: We do NOT require or auto-generate an `id` column.
    """
    df = df.copy()
    if col_map:
        df = df.rename(columns=col_map)
    # required base columns (no id)
    required_base = ["task","input_type","input"]
    for c in required_base:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if "question" not in df.columns:
        df["question"] = ""
    if "output" not in df.columns:
        df["output"] = ""

    # types (keep raw input for images if bytes)
    for c in ["task","input_type","question","output"]:
        df[c] = df[c].astype(str)

    # normalize input_type
    def _norm_it(x: str) -> str:
        x = str(x).lower().strip()
        if "img" in x or x == "image":
            return "image"
        return "text" if "text" in x or x == "text" else x
    df["input_type"] = df["input_type"].map(_norm_it)

    return df[["task","input_type","input","question","output"]]


def join_labels(inputs_df: pd.DataFrame, labels_df: pd.DataFrame,
                id_col_in_labels: str = "id", label_col_in_labels: str = "output") -> pd.DataFrame:
    labels_df = labels_df[[id_col_in_labels, label_col_in_labels]].copy()
    labels_df[id_col_in_labels] = labels_df[id_col_in_labels].astype(str)

    out = inputs_df.merge(labels_df, left_on="id", right_on=id_col_in_labels,
                          how="left", suffixes=("","_lab"))
    cond = (out["output"].isna()) | (out["output"].astype(str).str.len() == 0)
    out.loc[cond, "output"] = out.loc[cond, label_col_in_labels].astype(str)

    out = out.drop(columns=[id_col_in_labels, label_col_in_labels])
    out = out[~out["output"].isna() & (out["output"].astype(str).str.len() > 0)].reset_index(drop=True)
    return out


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
    print(f"[Build] out_root={out_root}")
    out_dir = os.path.join(out_root, DATA_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    inputs = read_any(inputs_file)
    inputs = to_unified_schema(inputs, col_map_inputs)

    if labels_file:
        labels = read_any(labels_file)
        if col_map_labels:
            labels = labels.rename(columns=col_map_labels)
        data = join_labels(inputs, labels, id_col_in_labels="id", label_col_in_labels="output")
    else:
        data = inputs
        data = data[data["output"].astype(str).str.len() > 0].reset_index(drop=True)

    train_df, valid_df = stratified_split(data, valid_ratio=valid_ratio, seed=42)

    def _stat(x: pd.DataFrame, name: str):
        print(f"\n[{name}] n={len(x)}")
        print(x.groupby(["task","input_type"]).size().sort_values(ascending=False).to_string())

    _stat(train_df, "train")
    _stat(valid_df, "valid")

    save_both_parquet_jsonl(train_df, os.path.join(out_dir, "train"))
    save_both_parquet_jsonl(valid_df, os.path.join(out_dir, "valid"))

    print(f"\n[Saved] {os.path.join(out_dir, 'train')}.(parquet|jsonl)")
    print(f"[Saved] {os.path.join(out_dir, 'valid')}.(parquet|jsonl)")
    return train_df, valid_df


# =============================
# Prompt builder (single path)
# =============================

def load_image_any(src) -> Image.Image:
    """Load PIL.Image from local path, URL, bytes, base64 data URI, numpy array, or PIL.Image."""
    # bytes / bytearray
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src)).convert("RGB")
    # base64 data URI string
    if isinstance(src, str) and src.startswith("data:image/"):
        try:
            header, b64 = src.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            pass
    # http(s) URL
    if isinstance(src, str) and (src.startswith("http://") or src.startswith("https://")):
        resp = requests.get(src, timeout=20)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    # numpy array
    try:
        import numpy as _np
        if isinstance(src, _np.ndarray):
            if src.dtype != _np.uint8:
                src = src.astype(_np.uint8)
            if src.ndim == 2:
                return Image.fromarray(src, "L").convert("RGB")
            return Image.fromarray(src)
    except Exception:
        pass
    # local path or already PIL
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    return Image.open(src).convert("RGB")


def build_messages(inp_type: str, task: str, inp: str, question: str, output: Optional[str] = None,
                   add_task_hint: bool = False) -> List[Dict[str, Any]]:
    system = {"role": "system", "content": SYSTEM_PROMPT}

    hint = f"[TASK={task}]\n" if add_task_hint else ""
    user_text_parts = ["Please produce the required output for the given input."]
    if question and str(question).strip():
        user_text_parts.append(f"Question:\n{str(question).strip()}")
    if inp_type == "text":
        user_text_parts.append(f"Input:\n{inp}")
        user = {"role": "user", "content": f"{hint}" + "\n".join(user_text_parts)}
    else:
        user = {"role": "user", "content": [
            {"type": "image", "image": inp},
            {"type": "text",  "text": f"{hint}" + "\n".join(user_text_parts)},
        ]}

    msgs = [system, user]
    if output is not None:
        msgs.append({"role": "assistant", "content": str(output).rstrip()})
    return msgs


def extract_and_load_images(messages: List[Dict[str, Any]]) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
    images: List[Image.Image] = []
    msgs = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            new_list = []
            for c in msg["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    img_src = c.get("image")
                    pil = img_src if isinstance(img_src, Image.Image) else load_image_any(img_src)
                    images.append(pil)
                    new_list.append({"type": "image", "image": pil})
                else:
                    new_list.append(c)
            msgs.append({"role": msg["role"], "content": new_list})
        else:
            msgs.append(msg)
    return images, msgs


# =============================
# Dataset with label masking
# =============================

class MMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: AutoProcessor, add_task_hint: bool = False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.add_task_hint = add_task_hint

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx].to_dict()
        inp_type = str(row.get("input_type")).strip().lower()
        task     = str(row.get("task")).strip()
        inp      = row.get("input")
        q        = row.get("question") or ""
        out      = str(row.get("output")).rstrip()

        # Build two message variants: prompt-only and with-answer
        msgs_prompt_only = build_messages(inp_type, task, inp, q, output=None, add_task_hint=self.add_task_hint)
        msgs_with_answer = build_messages(inp_type, task, inp, q, output=out, add_task_hint=self.add_task_hint)

        # Load images into messages
        images_p, msgs_prompt_only = extract_and_load_images(msgs_prompt_only)
        images_f, msgs_with_answer = extract_and_load_images(msgs_with_answer)
        # Sanity: same images list
        images = images_f if len(images_f) > 0 else None

        # Convert to chat template text (keep tokenize=False)
        prompt_text = self.processor.apply_chat_template(
            msgs_prompt_only, tokenize=False, add_generation_prompt=False
        )
        full_text = self.processor.apply_chat_template(
            msgs_with_answer, tokenize=False, add_generation_prompt=False
        )

        # Tokenize both; include images to keep token alignment consistent
        enc_full = self.processor(
            text=full_text,
            images=images,
            return_tensors="pt",
            padding="longest",
        )
        enc_prompt = self.processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
            padding="longest",
        )

        input_ids = enc_full["input_ids"][0]
        attn_mask = enc_full["attention_mask"][0]
        labels = input_ids.clone()
        prompt_len = enc_prompt["input_ids"].shape[-1]
        labels[:prompt_len] = -100  # mask prompt tokens

        item = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }
        # pass-through vision tensors if any
        for k in enc_full:
            if k not in item and isinstance(enc_full[k], torch.Tensor):
                item[k] = enc_full[k][0]
        return item


# =============================
# Collator
# =============================

@dataclass
class VLDataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Robustly collate mixed vision/text samples.
        - Pads input_ids/attention_mask/labels to the longest.
        - For optional vision keys (e.g., pixel_values, image_sizes, pixel_attention_mask, etc.),
          creates zeros-like placeholders for samples that don't have them so a mixed batch can stack.
        - If stacking still fails due to shape mismatch, falls back to a Python list for that key.
        """
        batch: Dict[str, Any] = {}

        # ---- 1) Text tensors (always present) ----
        text_keys = ["input_ids", "attention_mask", "labels"]
        for k in text_keys:
            if k not in features[0]:
                # If missing (shouldn't happen), skip
                continue
            max_len = max(f[k].shape[-1] for f in features)
            tensors = []
            for f in features:
                t = f[k]
                pad_len = max_len - t.shape[-1]
                if pad_len > 0:
                    pad_val = self.pad_token_id if k != "labels" else -100
                    t = F.pad(t, (0, pad_len), value=pad_val)
                tensors.append(t)
            batch[k] = torch.stack(tensors, dim=0)

        # ---- 2) Optional keys (union across features) ----
        # Collect union of keys minus text keys
        all_keys = set().union(*(f.keys() for f in features)) - set(text_keys)
        for k in sorted(all_keys):
            # Gather values (may include missing)
            vals = [f.get(k, None) for f in features]

            # Case A: tensors (some may be None)
            if any(isinstance(v, torch.Tensor) for v in vals):
                # choose a reference tensor shape/dtype
                ref = next((v for v in vals if isinstance(v, torch.Tensor)), None)
                if ref is None:
                    batch[k] = vals  # nothing to stack
                    continue
                # replace None with zeros-like
                filled = [v if isinstance(v, torch.Tensor) else torch.zeros_like(ref) for v in vals]
                try:
                    batch[k] = torch.stack(filled, dim=0)
                except Exception:
                    # last resort: keep as list (model should ignore unused vision keys)
                    batch[k] = filled
            else:
                # Non-tensor payloads (e.g., strings/lists); keep aligned list
                batch[k] = vals

        return batch


# =============================
# Training
# =============================

def make_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # recommended default
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, # A100-friendly
    )


def make_lora_config(r=64, alpha=128, dropout=0.05, target_modules: Optional[List[str]] = None) -> LoraConfig:
    if target_modules is None:
        target_modules = [
            "q_proj","k_proj","v_proj","o_proj",
            "up_proj","down_proj","gate_proj",
        ]  # language modules only (phase 1)
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
            gradient_accumulation_steps=16,  # eff 32
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
            gradient_accumulation_steps=8,  # eff 64
            learning_rate=1e-4,
            num_train_epochs=1,
            warmup_ratio=0.03,
            weight_decay=0.01,
            logging_steps=20,
            save_steps=1000,
            eval_steps=400,
        )
    # balanced (default)
    return dict(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # eff 32
        learning_rate=1e-4,
        num_train_epochs=1,
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=25,
        save_steps=1000,
        eval_steps=500,
    )


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

    # Load processor / tokenizer
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model 4-bit
    print(f"[Load] {base_model} in 4-bit (nf4, bfloat16 compute)")
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

    # LoRA (language modules only)
    peft_cfg = make_lora_config(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target_modules=target_modules)
    model = get_peft_model(model, peft_cfg)
    print("[LoRA] target modules:", peft_cfg.target_modules)

    # Datasets
    train_df = read_any(train_file)
    valid_df = read_any(valid_file) if valid_file else None

    required = {"task","input_type","input","question","output"}
    if not required.issubset(set(train_df.columns)):
        missing = required - set(train_df.columns)
        raise ValueError(f"Train file missing columns: {missing}")
    if valid_df is not None and not required.issubset(set(valid_df.columns)):
        missing = required - set(valid_df.columns)
        raise ValueError(f"Valid file missing columns: {missing}")

    train_ds = MMDataset(train_df, processor, add_task_hint=add_task_hint)
    eval_ds = MMDataset(valid_df, processor, add_task_hint=False) if valid_df is not None else None
    collator = VLDataCollator(pad_token_id=tokenizer.pad_token_id)

    # Training args (profile + common)
    p = resolve_profile(profile)
    print(f"[Profile] {profile} => {p}")

    # Build TrainingArguments with backward/forward compatibility
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
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    sig = inspect.signature(TrainingArguments.__init__)
    def maybe(key, value):
        if key in sig.parameters and value is not None:
            ta_kwargs[key] = value

    # evaluation-related
    eval_strategy = "steps" if eval_ds is not None else "no"
    maybe("evaluation_strategy", eval_strategy)
    maybe("eval_strategy", eval_strategy)  # some versions use this alias
    maybe("eval_steps", p.get("eval_steps") if eval_ds is not None else None)
    maybe("load_best_model_at_end", bool(eval_ds))
    maybe("metric_for_best_model", "eval_loss" if eval_ds is not None else None)
    maybe("greater_is_better", False if eval_ds is not None else None)

    # optimizer (fallback if not supported)
    maybe("optim", "paged_adamw_8bit")

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("[Train] start")
    trainer.train()
    print("[Train] done")

    # Save adapter + processor/tokenizer
    print(f"[Save] adapter to: {run_dir}")
    trainer.model.save_pretrained(run_dir)
    processor.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)

    print("[All done] Run dir:", run_dir)


# =============================
# Quick inference (optional)
# =============================

def quick_infer(run_dir: str, base_model: str = DEFAULT_BASE_MODEL, image_url: Optional[str] = None, question: Optional[str] = None):
    from peft import PeftModel
    try:
        model = AutoModelForVision2Seq.from_pretrained(base_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, run_dir)
    model.eval()

    processor = AutoProcessor.from_pretrained(run_dir, trust_remote_code=True)

    msgs = [{"role":"system","content":SYSTEM_PROMPT}]
    images = None
    if image_url:
        img = Image.open(io.BytesIO(requests.get(image_url, timeout=20).content)).convert("RGB")
        msgs.append({"role":"user","content":[{"type":"image","image":img},{"type":"text","text":question or "What is in the image?"}]})
        images = [img]
    else:
        msgs.append({"role":"user","content":question or "Please provide a short answer."})

    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    text = processor.decode(out[0], skip_special_tokens=True)
    print("\n[GEN]\n", text)


# =============================
# CLI
# =============================

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build train/valid datasets")
    p_build.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p_build.add_argument("--inputs_file", type=str, required=True)
    p_build.add_argument("--labels_file", type=str, default=None)
    p_build.add_argument("--valid_ratio", type=float, default=0.1)

    p_train = sub.add_parser("train", help="Train QLoRA adapter")
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
