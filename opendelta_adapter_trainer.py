# opendelta_adapter_trainer.py
# OpenDelta를 사용한 Qwen2.5-VL-7B Adapter Fine-tuning
# Requirements: pip install opendelta transformers>=4.55.2 bitsandbytes>=0.47.0

import os, io, re, json, base64, hashlib, warnings, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageFile

from transformers import (
    AutoModelForVision2Seq, AutoModelForCausalLM,
    AutoProcessor, AutoTokenizer,
    BitsAndBytesConfig, Trainer, TrainingArguments,
)

# OpenDelta imports
from opendelta import AdapterModel, LoraModel, AutoDeltaModel
from opendelta.auto_delta import AutoDeltaConfig

# ========= Safety / Env =========
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb=128")
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# ========= System prompt =========
SYSTEM_PROMPT = (
    "You are a single vision–language assistant. You will receive either an image or text, and optionally a question.\n"
    "Always reply in English in a single, unified style across tasks.\n"
    "If the answer cannot be determined from the provided input, output exactly: unknown."
)

# ========= Image I/O (simplified) =========
LONG_SIDE = 1280
MAX_SIDE_HARD = 3500
MIN_IMG_SIDE = 64
RESIZE_CACHE_DIR = "/tmp/img_resized_cache"
os.makedirs(RESIZE_CACHE_DIR, exist_ok=True)

def load_image_strict(input_obj) -> Image.Image:
    """로컬 경로 또는 Base64로부터 이미지 로드"""
    try:
        if isinstance(input_obj, str):
            # 로컬 파일 경로
            img = Image.open(input_obj).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(input_obj)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {input_obj!r} -> {e}")
    
    # 크기 제한
    if max(img.size) > MAX_SIDE_HARD:
        img.thumbnail((MAX_SIDE_HARD, MAX_SIDE_HARD), Image.LANCZOS)
    if max(img.size) > LONG_SIDE:
        img.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
    
    w, h = img.size
    if min(w, h) < MIN_IMG_SIDE:
        raise ValueError(f"Image too small: {w}x{h} < {MIN_IMG_SIDE}")
    return img

# ========= Prompt helpers =========
def build_user_prompt(task: str, input_type: str, the_input: str, question: Optional[str]) -> str:
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

def build_messages(inp_type: str, task: str, inp: str, question: str, output: Optional[str] = None, img_obj: Optional[Image.Image] = None):
    user_prompt = build_user_prompt(task, inp_type, str(inp), question)
    if inp_type == "image":
        img = img_obj if img_obj is not None else load_image_strict(inp)
        user_content = [{"type": "image"}, {"type": "text", "text": user_prompt}]
        images = [img]
    else:
        user_content = [{"type": "text", "text": user_prompt}]
        images = None

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": user_content},
    ]
    if output is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": str(output).rstrip()}]})
    return msgs, images

# ========= Dataset =========
class MMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: AutoProcessor):
        self.df = df.reset_index(drop=True)
        self.p = processor
        self.max_len = getattr(self.p.tokenizer, "model_max_length", 4096)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        inp_type = str(row["input_type"]).strip().lower()
        task     = str(row["task"])
        inp      = row["input"]
        q        = row.get("question", "") or ""
        out      = str(row.get("output", "")).rstrip()

        img_obj = None
        if inp_type == "image":
            img_obj = load_image_strict(inp)

        msgs_prompt, images = build_messages(inp_type, task, inp, q, output=None, img_obj=img_obj)
        msgs_full,   _      = build_messages(inp_type, task, inp, q, output=out, img_obj=img_obj)

        prompt_text = self.p.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=False)
        full_text   = self.p.apply_chat_template(msgs_full,   tokenize=False, add_generation_prompt=False)

        if images is not None:
            enc_full = self.p(text=full_text, images=images, return_tensors="pt", padding="longest")
            enc_prm  = self.p(text=prompt_text, images=images, return_tensors="pt", padding="longest")
        else:
            enc_full = self.p(text=full_text, return_tensors="pt", padding="longest",
                              truncation=True, max_length=self.max_len)
            enc_prm  = self.p(text=prompt_text, return_tensors="pt", padding="longest",
                              truncation=True, max_length=self.max_len)

        input_ids = enc_full["input_ids"][0]
        labels = input_ids.clone()
        prm_len = enc_prm["input_ids"].shape[-1]
        labels[:prm_len] = -100
        
        item = {
            "input_ids": input_ids,
            "attention_mask": enc_full["attention_mask"][0],
            "labels": labels,
        }
        
        pv = enc_full.get("pixel_values", None)
        if isinstance(pv, list):
            pv = pv[0] if len(pv) > 0 else None
        if isinstance(pv, torch.Tensor):
            if pv.dim() == 4 and pv.size(0) == 1:
                pv = pv.squeeze(0)
            if pv.dim() == 3 and pv.size(0) == 1:
                pv = pv.repeat(3, 1, 1)
            if pv.dim() == 3 or pv.dim() == 4:
                item["pixel_values"] = pv
        return item

@dataclass
class QwenVLDataCollator:
    pad_token_id: int = 0

    def __call__(self, features):
        input_ids      = [f["input_ids"]      for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels         = [f["labels"]         for f in features]
        max_len = max(x.size(0) for x in input_ids)

        def pad1d(x, val):
            if x.size(0) < max_len:
                pad = torch.full((max_len-x.size(0),), val, dtype=x.dtype, device=x.device)
                return torch.cat([x, pad], dim=0)
            return x

        batch = {
            "input_ids":      torch.stack([pad1d(x, self.pad_token_id) for x in input_ids]),
            "attention_mask": torch.stack([pad1d(x, 0)                 for x in attention_mask]),
            "labels":         torch.stack([pad1d(x, -100)              for x in labels]),
        }

        pvs = [f.get("pixel_values", None) for f in features]
        if any(isinstance(pv, torch.Tensor) for pv in pvs):
            norm = []
            Hs, Ws = [], []
            for pv in pvs:
                if isinstance(pv, torch.Tensor) and pv.dim() == 3:
                    c, h, w = pv.shape
                    if c == 1:
                        pv = pv.repeat(3, 1, 1)
                        c, h, w = pv.shape
                    norm.append(pv); Hs.append(h); Ws.append(w)
                else:
                    norm.append(None)
            if len(Hs) > 0:
                max_h, max_w = max(Hs), max(Ws)
                padded = []
                for pv in norm:
                    if pv is None:
                        pv = torch.zeros((3, max_h, max_w), dtype=batch["input_ids"].dtype)
                    else:
                        c, h, w = pv.shape
                        if h != max_h or w != max_w:
                            pv = F.pad(pv, (0, max_w-w, 0, max_h-h), value=0)
                    padded.append(pv)
                batch["pixel_values"] = torch.stack(padded)
        return batch

# ========= OpenDelta Adapter Configuration =========
def apply_adapter_to_model(model, adapter_type="adapter", bottleneck_dim=256, exclude_vision_backbone=True):
    """
    OpenDelta를 사용하여 모델에 Adapter 적용
    
    Args:
        model: Base model
        adapter_type: "adapter", "lora", or "prefix"
        bottleneck_dim: Adapter의 bottleneck dimension
        exclude_vision_backbone: Vision backbone 제외 여부
    """
    
    # Vision backbone을 제외할 모듈 패턴 설정
    if exclude_vision_backbone:
        # LLM 레이어만 선택 (vision 관련 제외)
        modified_modules = []
        for name, module in model.named_modules():
            # Vision backbone 제외
            if 'visual.blocks' in name:
                continue
            # LLM attention/mlp 레이어 선택
            if any(x in name for x in ['q_proj', 'v_proj', 'k_proj', 'o_proj']):
                if 'visual' not in name:  # LLM layers only
                    modified_modules.append(name)
            # Vision projector (merger) 포함
            elif 'visual.merger' in name:
                modified_modules.append(name)
    else:
        modified_modules = None  # 모든 레이어에 적용
    
    print(f"\n[INFO] Applying {adapter_type} to model")
    
    if adapter_type == "adapter":
        # Adapter 적용
        delta_model = AdapterModel(
            backbone_model=model,
            bottleneck_dim=bottleneck_dim,
            modified_modules=modified_modules,
            backend='torch'  # 'hf' or 'torch'
        )
        print(f"  - Bottleneck dimension: {bottleneck_dim}")
        
    elif adapter_type == "lora":
        # LoRA 적용 (비교용)
        delta_model = LoraModel(
            backbone_model=model,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            modified_modules=modified_modules,
            backend='torch'
        )
        print(f"  - LoRA rank: 16")
        
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    # 모듈 정보 출력
    if modified_modules:
        print(f"  - Modified modules: {len(modified_modules)}")
        print(f"  - Sample modules: {modified_modules[:5]}")
    
    # Delta 파라미터만 학습 가능하도록 설정
    delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
    
    # 파라미터 통계
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[INFO] Parameter Statistics:")
    print(f"  - Trainable: {trainable:,} ({trainable/1e6:.1f}M)")
    print(f"  - Total: {total:,} ({total/1e6:.1f}M)")
    print(f"  - Percentage: {100*trainable/total:.2f}%")
    
    return model, delta_model

# ========= BitsAndBytes config =========
def make_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

# ========= Training Function =========
def train(
    base_model: str,
    train_path: str,
    valid_path: Optional[str],
    out_dir: str,
    adapter_type: str = "adapter",  # "adapter" or "lora"
    bottleneck_dim: int = 256,      # Adapter bottleneck dimension
    profile: str = "base",
    max_steps: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    OpenDelta Adapter를 사용한 학습
    """
    # Set seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[CONFIG] OpenDelta {adapter_type.upper()} Training")
    print(f"{'='*60}")
    print(f"Base Model: {base_model}")
    print(f"Adapter Type: {adapter_type}")
    print(f"Bottleneck Dim: {bottleneck_dim}")
    print(f"Profile: {profile}")
    print(f"Output Dir: {out_dir}")
    print(f"{'='*60}\n")

    # Load processor/tokenizer
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with quantization
    bnb_cfg = make_bnb_config()
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            base_model, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_cfg, 
            trust_remote_code=True,
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_cfg, 
            trust_remote_code=True,
        )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Apply OpenDelta Adapter
    model, delta_model = apply_adapter_to_model(
        model, 
        adapter_type=adapter_type,
        bottleneck_dim=bottleneck_dim,
        exclude_vision_backbone=True
    )

    # Load data
    def read_any(path: str) -> pd.DataFrame:
        if path.endswith(".parquet"): 
            return pd.read_parquet(path)
        elif path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                return pd.DataFrame(json.loads(l) for l in f if l.strip())
        else:
            return pd.read_csv(path)
    
    train_df = read_any(train_path)
    valid_df = read_any(valid_path) if valid_path else None
    
    print(f"\n[DATA] Train samples: {len(train_df):,}")
    if valid_df is not None:
        print(f"[DATA] Valid samples: {len(valid_df):,}")
    
    # Create datasets
    train_ds = MMDataset(train_df, processor)
    eval_ds = MMDataset(valid_df, processor) if valid_df is not None else None
    collator = QwenVLDataCollator(pad_token_id=tokenizer.pad_token_id)

    # Training profiles
    profiles = {
        "dev": dict(
            num_train_epochs=3, 
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16, 
            learning_rate=5e-5,
            warmup_steps=100,
            logging_steps=50, 
            eval_steps=200, 
            save_steps=200,
        ),
        "base": dict(
            num_train_epochs=10, 
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16, 
            learning_rate=3e-5,
            warmup_steps=200,
            logging_steps=50, 
            eval_steps=500, 
            save_steps=500,
        ),
    }
    
    p = profiles.get(profile, profiles["base"])
    if max_steps:
        p["max_steps"] = max_steps
        p.pop("num_train_epochs", None)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        remove_unused_columns=False,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        save_total_limit=2,
        logging_dir=os.path.join(out_dir, "logs"),
        report_to="none",
        evaluation_strategy="steps" if eval_ds else "no",
        eval_strategy="steps" if eval_ds else "no",
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss" if eval_ds else None,
        greater_is_better=False if eval_ds else None,
        **p
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Train
    if resume_from_checkpoint:
        print(f"\n[INFO] Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    # Save final model and delta weights
    final_dir = os.path.join(out_dir, "final_adapter")
    os.makedirs(final_dir, exist_ok=True)
    
    # Save delta model state
    delta_model.save_finetuned(final_dir)
    
    # Save processor/tokenizer
    processor.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"\n[COMPLETED] Adapter saved to: {final_dir}")
    print(f"[COMPLETED] Processor/Tokenizer saved to: {out_dir}")
    
    return final_dir

# ========= CLI =========
def build_parser():
    p = argparse.ArgumentParser(description="Qwen2.5-VL OpenDelta Adapter Training")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--valid_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--adapter_type", choices=["adapter", "lora"], default="adapter")
    p.add_argument("--bottleneck_dim", type=int, default=256)
    p.add_argument("--profile", choices=["dev","base"], default="base")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p

def main():
    args = build_parser().parse_args()
    train(
        base_model=args.base_model,
        train_path=args.train_path,
        valid_path=args.valid_path,
        out_dir=args.out_dir,
        adapter_type=args.adapter_type,
        bottleneck_dim=args.bottleneck_dim,
        profile=args.profile,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

if __name__ == "__main__":
    main()
