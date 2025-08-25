# qwen25_vl_adapter_finetune/trainer.py
# transformers==4.55.2, bitsandbytes==0.47.0
# Qwen2.5-VL-7B — Unified Multitask Fine-tuning (Adapter)
import os, io, re, csv, json, base64, hashlib, warnings, argparse, inspect, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms

from transformers import (
    AutoModelForVision2Seq, AutoModelForCausalLM,
    AutoProcessor, AutoTokenizer,
    BitsAndBytesConfig, Trainer, TrainingArguments,
)

from peft import AdapterConfig, get_peft_model, prepare_model_for_kbit_training

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

# ========= Image I/O =========
from urllib.parse import urlparse
#from urllib.request import urlopen, Request

URL_TIMEOUT = 20
LONG_SIDE = 1280
MAX_SIDE_HARD = 3500
MIN_IMG_SIDE = 64
RESIZE_CACHE_DIR = "/tmp/img_resized_cache"
os.makedirs(RESIZE_CACHE_DIR, exist_ok=True)

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
    img.thumbnail((cap, cap), Image.LANCZOS)
    return img

def _resize_keep_ratio(img: Image.Image, long_side: int) -> Image.Image:
    if max(img.size) <= long_side:
        return img
    img = img.copy()
    img.thumbnail((long_side, long_side), Image.LANCZOS)
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
    """
    로컬 경로, bytes, Base64로부터 이미지를 로드합니다.
    URL 로딩 기능은 제거되었습니다.
    """
    try:
        if isinstance(input_obj, (bytes, bytearray)):
            # 1. raw bytes 데이터 처리
            img = Image.open(io.BytesIO(input_obj)).convert("RGB")
        elif isinstance(input_obj, str):
            s = input_obj.strip()
            if s.startswith("data:image"):
                # 2. Data URL 형식의 Base64 처리
                b64 = s.split(",", 1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            elif looks_like_base64(s):
                # 3. 순수 Base64 문자열 처리
                img = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
            else:
                # 4. 로컬 파일 경로 처리 (URL 확인 로직 없음)
                #    이제 `input` 열에 있는 다운로드된 이미지의 경로가 여기에 해당됩니다.
                img = Image.open(s).convert("RGB")
        else:
            raise TypeError(f"지원하지 않는 이미지 입력 타입: {type(input_obj)}")
    except Exception as e:
        # 파일이 없거나 손상된 경우 에러 발생
        raise RuntimeError(f"[ImageLoadError] 이미지를 불러오는 데 실패했습니다: {input_obj!r} -> {e}")

    # 이미지 리사이즈 및 최종 처리
    img = finalize_image(img)
    w, h = img.size
    if min(w, h) < MIN_IMG_SIDE:
        raise ValueError(f"[ImageTooSmall] 이미지 크기가 너무 작습니다: {w}x{h} < {MIN_IMG_SIDE}")
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

# ========= Dataset & Collator =========
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

        # 이미지 처리
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
        
        # pixel_values 정규화
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
        # ---- 텍스트 패딩 ----
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

        # ---- 픽셀 패딩 (3D/None) ----
        pvs = [f.get("pixel_values", None) for f in features]
        if any(isinstance(pv, torch.Tensor) for pv in pvs):
            # 유효한 3D만 취급, 나머지는 None
            norm = []
            Hs, Ws = [], []
            for pv in pvs:
                if isinstance(pv, torch.Tensor) and pv.dim() == 3:
                    c, h, w = pv.shape
                    # 채널 보정(혹시 c==1로 올 수도 있음)
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
                batch["pixel_values"] = torch.stack(padded)  # (B, C, H, W)

        return batch

# ========= Adapter config =========
def make_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def make_adapter_config(reduction_factor=16, adapter_dropout=0.1) -> AdapterConfig:
    """Standard Adapter configuration"""
    return AdapterConfig(
        reduction_factor=reduction_factor,  # Hidden dimension = input_dim / reduction_factor
        non_linearity="relu",  # Activation function
        adapter_dropout=adapter_dropout,
        target_modules=["q_proj", "v_proj"],  # Where to insert adapters
        scaling=1.0,  # Scaling factor for adapter output
    )

def make_adapter_config_with_projector(model, reduction_factor=16, adapter_dropout=0.1) -> AdapterConfig:
    """Adapter configuration for LLM + Vision Projector"""
    
    # Adapter를 삽입할 모듈 찾기
    target_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Vision Backbone 제외
            if 'visual.blocks' in name:
                continue
            
            # LLM attention layers (q_proj, v_proj만 사용 - Adapter 관례)
            if 'visual' not in name:
                if any(x in name for x in ['q_proj', 'v_proj']):
                    # 전체 경로 추가
                    target_modules.append(name)
            
            # Vision Projector layers
            elif 'visual.merger' in name:
                target_modules.append(name)
    
    print(f"\n[INFO] Selected {len(target_modules)} modules for Adapters")
    print("[INFO] Module distribution:")
    
    # 모듈 분포 확인
    llm_count = sum(1 for m in target_modules if 'visual' not in m)
    projector_count = sum(1 for m in target_modules if 'visual.merger' in m)
    
    print(f"  - LLM modules: {llm_count}")
    print(f"  - Projector modules: {projector_count}")
    
    # Vision backbone 체크
    backbone_modules = [m for m in target_modules if 'visual.blocks' in m]
    if backbone_modules:
        print(f"\n⚠️ ERROR: {len(backbone_modules)} backbone modules incorrectly included!")
    else:
        print("  ✓ Vision Backbone successfully excluded")
    
    print("\n[INFO] Sample target modules:")
    for m in target_modules[:5]:
        print(f"  - {m}")
    
    return AdapterConfig(
        reduction_factor=reduction_factor,
        non_linearity="relu",
        adapter_dropout=adapter_dropout,
        target_modules=target_modules,
        scaling=1.0,
    )

def fix_trainer_state_json(checkpoint_path: str):
    """trainer_state.json에서 호환되지 않는 필드 제거"""
    state_path = os.path.join(checkpoint_path, "trainer_state.json")
    
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        # 호환되지 않는 필드 제거
        incompatible_fields = ['learning_rate', 'eval_steps', 'save_steps']
        modified = False
        
        for field in incompatible_fields:
            if field in state:
                del state[field]
                modified = True
                print(f"[INFO] Removed incompatible field '{field}' from trainer_state.json")
        
        if modified:
            # 백업 생성
            backup_path = state_path + '.backup'
            with open(backup_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            # 수정된 파일 저장
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"[INFO] Updated trainer_state.json (backup saved as {backup_path})")
    
    return state_path
  
# ========= Utilities =========
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"): return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.loads(l) for l in f if l.strip())
    return pd.read_csv(path)

# ========= Train (4.55.2-safe) =========
# ========= Train (4.55.2-safe) =========
def train(
    base_model: str,
    train_path: str,
    valid_path: Optional[str],
    out_dir: str,
    profile: str = "base",          # "dev" | "base" | "long"
    reduction_factor: int = 16,     # Adapter hidden dimension factor
    adapter_dropout: float = 0.1,   # Adapter dropout
    # 추가 파라미터들
    max_steps: Optional[int] = None,  # 최대 학습 스텝
    resume_from_checkpoint: Optional[str] = None,  # 체크포인트 재개
):
    set_seed(42)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[CONFIG] Training Configuration")
    print(f"{'='*60}")
    print(f"Profile: {profile}")
    print(f"Adapter Reduction Factor: {reduction_factor}")
    print(f"Adapter Dropout: {adapter_dropout}")
    print(f"Max Steps: {max_steps if max_steps else 'Not set (use epochs)'}")
    print(f"Resume from: {resume_from_checkpoint if resume_from_checkpoint else 'Fresh start'}")
    print(f"{'='*60}\n")

    # Processor/tokenizer
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model (Vision2Seq -> CausalLM fallback)
    bnb_cfg = make_bnb_config()
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            base_model, device_map="auto", torch_dtype=torch.bfloat16,
            quantization_config=bnb_cfg, trust_remote_code=True,
        )
    except Exception as e1:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto", torch_dtype=torch.bfloat16,
                quantization_config=bnb_cfg, trust_remote_code=True,
            )
        except Exception as e2:
            raise RuntimeError(f"Failed to load base model: {e1} / {e2}")

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
  
    print("\n[INFO] Configuring Adapters for LLM + Projector (Vision Backbone FROZEN)")
    adapter_cfg = make_adapter_config_with_projector(model, reduction_factor=reduction_factor, adapter_dropout=adapter_dropout)
    model = get_peft_model(model, adapter_cfg)
    
    # 확인
    print("\n[INFO] Verifying trainable parameters:")
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'visual.blocks' in name:
                print(f"  ⚠️ WARNING: Vision backbone is trainable: {name}")
            trainable += param.numel()
        else:
            frozen += param.numel()
    
    print(f"  - Trainable: {trainable:,} ({trainable/1e6:.1f}M)")
    print(f"  - Frozen: {frozen:,} ({frozen/1e6:.1f}M)")
    print(f"  - Percentage: {100*trainable/(trainable+frozen):.2f}%")
    
    # Data
    train_df = read_any(train_path)
    valid_df = read_any(valid_path) if valid_path else None
    need = {"task","input_type","input","question","output"}
    if not need.issubset(set(train_df.columns)):
        raise ValueError(f"Train file missing columns: {need - set(train_df.columns)}")
    if valid_df is not None and not valid_df.empty and not need.issubset(set(valid_df.columns)):
        raise ValueError(f"Valid file missing columns: {need - set(valid_df.columns)}")
    print(f"\n[DATA] Train samples: {len(train_df):,}")
    if valid_df is not None and not valid_df.empty:
        print(f"[DATA] Valid samples: {len(valid_df):,}")
    
    # 데이터셋 생성
    train_ds = MMDataset(train_df, processor)
    eval_ds = None
    if valid_df is not None and not valid_df.empty:
        eval_ds = MMDataset(valid_df, processor)
    collator = QwenVLDataCollator(pad_token_id=tokenizer.pad_token_id)

    profiles = {
        "dev": dict(num_train_epochs=10, per_device_train_batch_size=1, per_device_eval_batch_size=1,
                     gradient_accumulation_steps=16, learning_rate=3e-5, warmup_steps=200,
                     logging_steps=50, eval_steps=500, save_steps=500,  weight_decay=0.01,
                     max_grad_norm=1.0, load_best_model_at_end=True, metric_for_best_model="eval_loss", 
                     greater_is_better=False),
        "base": dict(num_train_epochs=10, per_device_train_batch_size=1, per_device_eval_batch_size=1,
                     gradient_accumulation_steps=16, learning_rate=5e-5, warmup_steps=200,
                     logging_steps=50, eval_steps=500, save_steps=500,  weight_decay=0.1,
                     max_grad_norm=1.0, load_best_model_at_end=True, metric_for_best_model="eval_loss", 
                     greater_is_better=False)
    }
    p = profiles.get(profile, profiles["base"])
    
    # Build TrainingArguments with signature-check (4.55.2 safe)
    ta = dict(
        output_dir=out_dir,
        remove_unused_columns=False,
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        save_total_limit=2,
        logging_dir=os.path.join(out_dir, "logs"),
        report_to="none",
        **p,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    def maybe_set(k, v):
        if k in sig.parameters and v is not None:
            ta[k] = v

    eval_strategy = "steps" if (valid_df is not None and not valid_df.empty) else "no"
    maybe_set("evaluation_strategy", eval_strategy)
    maybe_set("eval_strategy", eval_strategy)
    maybe_set("load_best_model_at_end", bool(valid_df is not None and not valid_df.empty))
    maybe_set("metric_for_best_model", "eval_loss" if (valid_df is not None and not valid_df.empty) else None)
    maybe_set("greater_is_better", False if (valid_df is not None and not valid_df.empty) else None)
    
    train_args = TrainingArguments(**ta)
    
    # 디버깅: TrainingArguments 객체에서 실제 값 확인
    print(f"\n[DEBUG] TrainingArguments loaded values:")
    print(f"  - eval_steps: {train_args.eval_steps}")
    print(f"  - save_steps: {train_args.save_steps}")
    print(f"  - eval_strategy: {getattr(train_args, 'eval_strategy', 'NOT SET')}")
    print(f"  - logging_steps: {train_args.logging_steps}\n")
    
    # 학습 시작 전 최종 확인
    input("Press Enter to continue with training...")  # 수동 확인용
    
    init_kwargs = dict(
        model=model, args=train_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
    )
    # tokenizer/processing_class compatibility
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        init_kwargs["processing_class"] = processor
    else:
        init_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**init_kwargs)
    
    # 초기 validation loss 확인
    if eval_ds is not None:
        initial_eval = trainer.evaluate()
        print(f"\n[INITIAL] Validation Loss: {initial_eval['eval_loss']:.4f}")
        
        # Loss가 너무 높으면 경고
        if initial_eval['eval_loss'] > 2.0:
            print("⚠️ WARNING: Initial validation loss is very high!")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Training aborted.")
                return None
                
    if resume_from_checkpoint:
        print(f"\n[INFO] Resuming from checkpoint: {resume_from_checkpoint}")
        
        # trainer_state.json 호환성 수정
        fix_trainer_state_json(resume_from_checkpoint)
        
        # adapter_config.json 확인
        adapter_config_path = os.path.join(resume_from_checkpoint, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
                old_modules = set(config.get('target_modules', []))
                new_modules = set(adapter_cfg.target_modules)
                
                if old_modules != new_modules:
                    print(f"\n⚠️ WARNING: Module configuration mismatch detected!")
                    print(f"  Old modules: {sorted(old_modules)}")
                    print(f"  New modules: {sorted(new_modules)}")
                    print("\nThis is expected if switching from LoRA to Adapter or changing target modules.")
                    print("The training will continue with the new configuration.\n")
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
      
    # Save adapter + processor/tokenizer
    adapter_dir = os.path.join(out_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"[Saved] adapter => {adapter_dir}")
    print(f"[Saved] processor/tokenizer => {out_dir}")
    return adapter_dir

# ========= CLI (optional) =========
def build_parser():
    p = argparse.ArgumentParser(description="Qwen2.5-VL-7B unified multitask fine-tuning (Adapter)")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--valid_path", type=str, default=None)
    p.add_argument("--out_dir",   type=str, required=True)
    p.add_argument("--profile",   choices=["dev","base","long"], default="base")
    p.add_argument("--reduction_factor", type=int, default=16, help="Adapter reduction factor")
    p.add_argument("--adapter_dropout", type=float, default=0.1, help="Adapter dropout")
    # 추가 arguments
    p.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    return p

def main():
    args = build_parser().parse_args()
    train(
        base_model=args.base_model,
        train_path=args.train_path,
        valid_path=args.valid_path,
        out_dir=args.out_dir,
        profile=args.profile,
        reduction_factor=args.reduction_factor,
        adapter_dropout=args.adapter_dropout,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

if __name__ == "__main__":
    main()
