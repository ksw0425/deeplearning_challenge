# ==========================
# Cell 1: Imports & Safety
# ==========================
import os, io, re, base64, hashlib, warnings
import pandas as pd
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError, ImageFile
from urllib.parse import urlparse
from urllib.request import urlopen, Request
import csv, json
from pathlib import Path
import torch
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# PIL safety knobs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=128"
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# ==========================
# Cell 2: CONFIG & SYSTEM PROMPT
# ==========================

# --- 모델 및 경로 설정 ---
BASE_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# LLM+Projector fine-tuning 결과 load
FT_OUT_DIR = "/content/drive/MyDrive/Colab Notebooks/wook/fine-tuning/datasets/efficient_lora/lora-out"
# 최종 체크포인트 번호로 변경 필요 (예: checkpoint-5000)
ADAPTER_DIR = f"{FT_OUT_DIR}/checkpoint-7000"  # 또는 checkpoint-XXXX
PROC_DIR = FT_OUT_DIR

DTYPE = torch.bfloat16

# 테스트 데이터 경로
TEST_PATH = "/content/drive/MyDrive/Colab Notebooks/wook/deeplearningchallenge/deep_chal_multitask_dataset_test.parquet"
OUT_DIR = "/content/drive/MyDrive/Colab Notebooks/wook/output"
IMG_BASE = "/content"
URL_TIMEOUT = 5
MAX_SIDE_HARD = 3500

# --- 이미지 및 생성 설정 ---
LONG_SIDE = 1280
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.9

# --- 시스템 프롬프트 ---
SYSTEM_PROMPT = """
You are a unified vision–language inference assistant for five tasks:
[captioning, vqa, summarization, text_qa, math_reasoning].

Always reply in English and output ONLY the answer according to task-specific rules.
If the answer cannot be determined from the provided input, output exactly: unknown.

Task-specific output rules:

**captioning**:
- MUST start with "The image is" or "The image shows"
- Provide a detailed description in 1-3 sentences (typically 100-200 words)
- Describe visual elements systematically: main subjects, colors, composition, background, style
- For book/magazine covers: mention title, author, publisher, visual elements, and any text visible
- For illustrations: describe the style (e.g., "digital illustration", "vintage", "cartoon-like")
- Include spatial relationships (e.g., "in the center", "on the right side", "in the background")
- End with overall impression if relevant (e.g., "The image has a futuristic and sci-fi feel to it")
- Example: "The image is the cover of a book titled... The cover features..."

**vqa**:
- Output ONLY the extracted text or answer from the image
- For text extraction: output exactly as shown (preserve capitalization, punctuation)
- For yes/no questions: use lowercase 'yes' or 'no'
- For counting: use digits only
- For names/titles: preserve exact formatting
- No additional words or explanations

**math_reasoning**:
- Show step-by-step calculation using the format: value = <<calculation>>result
- Each line should explain one step
- Use the exact format for calculations: <<operation>>
- End with: #### [final_answer]
- Example format:
  Mimi has 2 x 12 = <<2*12=24>>24 sea shells.
  Kyle has 24 x 2 = <<24*2=48>>48 sea shells.
  Leigh has 48 / 3 = <<48/3=16>>16 sea shells.
  #### 16

**summarization**:
- Single paragraph of 100-200 words
- For legislation: Start with act name, then main provisions
- Use format: "[Act Name] - [Main action verb] [key provisions]"
- Include specific requirements, timelines, and responsible parties
- Maintain neutral, factual tone
- No quotes or section numbers

**text_qa**:
- Return a JSON object with exactly these keys:
  {'input_text': [list of answers], 'answer_start': [list of start positions], 'answer_end': [list of end positions]}
- Extract answers directly from the source text
- For multiple questions, maintain order correspondence
- Use exact text spans from the passage
- Format: valid JSON without any additional text

Do not add any explanations, labels, or metadata beyond the specified format for each task.
""".strip()
print("Configuration and System Prompt are set.")
print(f"Using adapter from: {ADAPTER_DIR}")

# ==========================
# Cell 3: Helper Functions
# ==========================

def safe_to_csv_utf8(df: pd.DataFrame, path: str):
    out = df.copy()
    out.columns = [c.replace("\ufeff","").strip().lower() for c in out.columns]
    assert out.columns.tolist() == ["id", "output"], f"Columns must be ['id','output'], got {out.columns.tolist()}"
    out["id"] = out["id"].astype(str)
    out["output"] = out["output"].astype(str)
    kwargs = dict(index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    try:
        out.to_csv(path, lineterminator="\n", **kwargs)
    except TypeError:
        out.to_csv(path, **kwargs)

def assert_submission_utf8_ok(path: str):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise AssertionError(f"Not UTF-8: {e}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    if [h.strip().lower() for h in header] != ["id","output"]:
        raise AssertionError(f"Header mismatch: {header}")
    print(f"Submission '{path}' looks OK. ({len(rows)-1} rows)")

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

def _cap_max_side(img: Image.Image, cap=MAX_SIDE_HARD) -> Image.Image:
    if max(img.size) <= cap: return img
    img = img.copy()
    img.thumbnail((cap, cap), Image.LANCZOS)
    return img

def _resize_keep_ratio(img: Image.Image, long_side: int) -> Image.Image:
    if max(img.size) <= long_side: return img
    img = img.copy()
    img.thumbnail((long_side, long_side), Image.LANCZOS)
    return img

RESIZE_CACHE_DIR = "/tmp/img_resized_cache"
os.makedirs(RESIZE_CACHE_DIR, exist_ok=True)

def _hash_image_bytes(img: Image.Image) -> str:
    with io.BytesIO() as bio:
        img.save(bio, format="PNG", optimize=False)
        return hashlib.md5(bio.getvalue()).hexdigest()

def finalize_image(img: Image.Image) -> Image.Image:
    img = _cap_max_side(img, MAX_SIDE_HARD)
    target = LONG_SIDE
    if max(img.size) <= target: return img
    h = _hash_image_bytes(img) + f"_{target}"
    path = os.path.join(RESIZE_CACHE_DIR, h + ".png")
    if os.path.exists(path): return Image.open(path).convert("RGB")
    out = _resize_keep_ratio(img, target)
    out.save(path, format="PNG")
    return out

def load_image(input_obj) -> Image.Image:
    try:
        if isinstance(input_obj, (bytes, bytearray)):
            return _cap_max_side(Image.open(io.BytesIO(input_obj)).convert("RGB"))
        if isinstance(input_obj, str):
            if input_obj.startswith("data:image"):
                b64 = input_obj.split(",", 1)[1]
                return _cap_max_side(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
            if looks_like_base64(input_obj):
                return _cap_max_side(Image.open(io.BytesIO(base64.b64decode(input_obj))).convert("RGB"))
            p = str(input_obj)
            if is_url(p):
                req = Request(p, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=URL_TIMEOUT) as r:
                    raw = r.read()
                return _cap_max_side(Image.open(io.BytesIO(raw)).convert("RGB"))
            if not os.path.isabs(p) and IMG_BASE:
                p = os.path.join(IMG_BASE, p)
            return _cap_max_side(Image.open(p).convert("RGB"))
    except Exception:
        return Image.new("RGB", (1, 1), (0, 0, 0))
    return Image.new("RGB", (1, 1), (0, 0, 0))

print("Helper functions are defined.")

# ==========================
# Cell 4: Load Model & Processor
# ==========================
model, proc = None, None
if 'model' not in locals() or model is None:
    print(f"Loading base model '{BASE_ID}'...")
    
    # 학습 시와 동일한 방식으로 base model 로드
    try:
        base = AutoModelForVision2Seq.from_pretrained(
            BASE_ID, 
            device_map="auto", 
            trust_remote_code=True, 
            torch_dtype=DTYPE
        )
        print("Loaded as Vision2Seq model")
    except Exception as e1:
        try:
            base = AutoModelForCausalLM.from_pretrained(
                BASE_ID, 
                device_map="auto", 
                trust_remote_code=True, 
                torch_dtype=DTYPE
            )
            print("Loaded as CausalLM model")
        except Exception as e2:
            raise RuntimeError(f"Failed to load base model: {e1} / {e2}")
    
    # Processor는 fine-tuned 디렉토리에서 로드 (학습 시 저장된 것)
    print(f"Loading processor from '{PROC_DIR}'...")
    proc = AutoProcessor.from_pretrained(PROC_DIR, trust_remote_code=True)
    
    # Adapter 로드
    if ADAPTER_DIR and os.path.exists(ADAPTER_DIR):
        print(f"Loading adapter from '{ADAPTER_DIR}'...")
        model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
        print("Adapter loaded successfully")
    else:
        print(f"Warning: Adapter directory '{ADAPTER_DIR}' not found, using base model only")
        model = base
    
    model = model.eval()
    print("Model and processor loaded and set to evaluation mode.")
else:
    print("Model and processor are already loaded.")

# ==========================
# Cell 5: Inference Function
# ==========================
@torch.no_grad()
def infer_one(model, proc, task: str, input_type: str, the_input, question: str = "") -> str:
    user_prompt = build_user_prompt(task, input_type, str(the_input), question)

    if (input_type or "text").lower() == "image":
        img = load_image(the_input)
        img = finalize_image(img)
        user_content = [{"type": "image"}, {"type": "text", "text": user_prompt}]
        images = [img]
    else:
        user_content = [{"type": "text", "text": user_prompt}]
        images = None

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": user_content},
    ]

    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = dict(text=[text], return_tensors="pt", padding=True)
    if images is not None:
        kwargs["images"] = images
    
    batch = proc(**kwargs)
    device = next(model.parameters()).device
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    input_len = batch["input_ids"].shape[1]
    out_ids = model.generate(
        **batch,
        do_sample=bool(TEMPERATURE and TEMPERATURE > 0.0),
        temperature=TEMPERATURE, 
        top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=proc.tokenizer.eos_token_id,
        eos_token_id=proc.tokenizer.eos_token_id,
    )
    gen_only = out_ids[:, input_len:]
    out = proc.batch_decode(gen_only, skip_special_tokens=True)[0]
    return out.strip()

print("Inference function is defined.")

# ==========================
# Cell 6: Load Data & Run Inference
# ==========================
print(f"\nLoading test data from '{TEST_PATH}'...")
df = pd.read_parquet(TEST_PATH)

# Handle column name typo if exists
if "input_type" not in df.columns and "input_tpye" in df.columns:
    df = df.rename(columns={"input_tpye": "input_type"})
    print("Fixed column name 'input_tpye' -> 'input_type'")

df = df.reset_index(drop=True)
df.insert(0, "id", df.index.astype(str))

print(f"Total rows to process: {len(df):,}")
print(f"Tasks: {df['task'].value_counts().to_dict()}\n")

# Run inference
results = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
    output = infer_one(
        model,
        proc,
        task=row["task"],
        input_type=row["input_type"],
        the_input=row["input"],
        question=row.get("question", "")
    )
    results.append({"id": str(row["id"]), "output": output})
    
    # Print sample outputs for debugging (first 3)
    if i < 3:
        print(f"\nSample {i}:")
        print(f"  Task: {row['task']}")
        print(f"  Output preview: {output[:100]}...")

submission_df = pd.DataFrame(results)
print("\n✓ Inference complete.")

# ==========================
# Cell 7: Save and Verify Submission
# ==========================
os.makedirs(OUT_DIR, exist_ok=True)
submission_path = os.path.join(OUT_DIR, "submission.csv")

print(f"\nSaving submission to '{submission_path}'...")
safe_to_csv_utf8(submission_df, submission_path)

try:
    assert_submission_utf8_ok(submission_path)
    print("✓ Submission file verification passed")
except AssertionError as e:
    print(f"⚠ Verification failed: {e}")

print(f"\n✅ Submission file saved to: {submission_path}")
print(f"Total predictions: {len(submission_df)}")
