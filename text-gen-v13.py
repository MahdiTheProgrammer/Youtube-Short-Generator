"""
YouTube Shorts Generator with Local LLaMA (v2)
- Generates 300–500 word narration scripts (ready for TTS pipeline)
- Prevents repetition via embedding similarity and topic cooldown
- Produces 10 image prompts for each topic (cleaned/sanitized)
- Outputs also in variable assignment format (story_n / pic_prompts_n)

Dependencies:
    pip install transformers accelerate wikipedia-api sentence-transformers scikit-learn
"""

import os
import sqlite3
import pathlib
import textwrap
from datetime import datetime, timedelta
import random
import hashlib
import re
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import wikipediaapi
from sentence_transformers import SentenceTransformer, util

# =========================
# Config
# =========================
MODEL_ID = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.8
TOP_P = 0.9

OUTPUT_DIR = "outputs"
DB_PATH = "memory.sqlite"
EMB_MODEL = "all-MiniLM-L6-v2"
PAST_K = 50
SIM_THRESHOLD = 0.86
MIN_WORDS = 300
MAX_WORDS = 500
TOPIC_COOLDOWN_DAYS = 30

# Expanded pool of diverse topics
SEED_TOPICS = [
    "Black holes", "Pompeii", "Tardigrades", "Hedy Lamarr", "Sagrada Família",
    "Antikythera mechanism", "Roanoke Colony", "Voyager 1", "Quokka", "Fibonacci",
    "Neon", "Basilisk", "Great Barrier Reef", "Quantum entanglement", "Library of Alexandria",
    "Stonehenge", "Machu Picchu", "Great Wall of China", "Terracotta Army", "Shackleton Expedition",
    "Rosetta Stone", "Eiffel Tower", "Mona Lisa", "Vincent van Gogh", "Leonardo da Vinci",
    "Nikola Tesla", "Marie Curie", "Albert Einstein", "Isaac Newton", "Charles Darwin",
    "Periodic Table", "DNA double helix", "CRISPR", "Apollo 11", "Saturn V",
    "Mars Rover Perseverance", "James Webb Space Telescope", "Andromeda Galaxy", "Supernova",
    "Aurora Borealis", "Volcano Krakatoa", "Mount Everest", "Sahara Desert", "Amazon Rainforest",
    "Giant Squid", "Komodo Dragon", "Platypus", "Axolotl", "Okapi", "Narwhal",
    "Dodo bird", "Passenger pigeon", "Woolly mammoth", "Saber-toothed cat",
    "Silk Road", "Roman Empire", "Maya Civilization", "Aztec Empire", "Inca Empire",
    "Vikings", "Samurai", "Spartans", "Medieval Knights", "Crusades",
    "Printing Press", "Steam Engine", "Light Bulb", "Telephone", "Internet",
    "Artificial Intelligence", "Blockchain", "Quantum Computing"
]

GEN_PROMPT = (
    "You write YouTube Shorts narration scripts between 300 and 500 words.\n"
    "Use ONLY the provided facts; do not copy sentences from sources.\n"
    "Start with a 1-sentence hook that creates curiosity.\n"
    "Keep sentences short for TTS. Include exactly one twist.\n"
    "End with a 1-sentence punch.\n"
    "Write only the narration text. Do not include lists, headings, or meta notes."
)

REWRITE_PROMPT = (
    "Rewrite the script with a different storytelling angle and vocabulary, "
    "keeping the same facts. Expand if under 300 words. Keep it between 300 and 500 words. "
    "New hook, new twist, new punch. Write only narration text."
)

IMAGE_PROMPT_SYSTEM = (
    "Return ONLY a numbered list from 1 to 10 of image prompts. "
    "Each line MUST contain a single prompt. "
    "No preface, no system/user/assistant text, no extra lines."
)

# =========================
# Setup
# =========================
def ensure_dirs():
    pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True)

def connect_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS scripts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        topic TEXT,
        text TEXT
    )""")
    con.commit()
    return con

def get_recent_texts(con, k=PAST_K):
    cur = con.cursor()
    cur.execute("SELECT text FROM scripts ORDER BY id DESC LIMIT ?", (k,))
    return [r[0] for r in cur.fetchall()]

def get_recent_topics(con, days=TOPIC_COOLDOWN_DAYS):
    cutoff = datetime.utcnow() - timedelta(days=days)
    cur = con.cursor()
    cur.execute("SELECT topic FROM scripts WHERE created_at > ?", (cutoff.isoformat(),))
    return {r[0] for r in cur.fetchall()}

def save_script(con, topic, text):
    cur = con.cursor()
    cur.execute(
        "INSERT INTO scripts(created_at, topic, text) VALUES(?,?,?)",
        (datetime.utcnow().isoformat(), topic, text),
    )
    con.commit()

def save_to_file(topic, narration, img_prompts, idx=1):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base = f"{ts}_{hashlib.sha256(topic.encode()).hexdigest()[:10]}"
    narr_path = os.path.join(OUTPUT_DIR, f"{base}_narration.txt")
    img_path = os.path.join(OUTPUT_DIR, f"{base}_images.txt")
    # Only save the sanitized narration
    narration = sanitize_narration(narration)
    with open(narr_path, "w", encoding="utf-8") as f:
        f.write(narration)
    with open(img_path, "w", encoding="utf-8") as f:
        f.write(format_image_prompt_list("images_prompt", img_prompts))
    return narr_path, img_path

# =========================
# Wikipedia RAG helpers
# =========================
def pick_topic(wiki, con):
    recent = get_recent_topics(con)
    random.shuffle(SEED_TOPICS)
    for t in SEED_TOPICS:
        if t not in recent and wiki.page(t).exists():
            return t
    return random.choice(SEED_TOPICS)

def wiki_passages(wiki, title, max_passages=5):
    page = wiki.page(title)
    if not page.exists():
        return []
    chunks = []
    if page.summary:
        chunks.append(page.summary)
    for section in page.sections:
        if len(chunks) >= max_passages:
            break
        text = section.text
        if 350 < len(text) < 1200:
            chunks.append(text)
    return chunks[:max_passages]

def build_fact_block(passages, limit_chars=1600):
    combined = []
    total = 0
    for p in passages:
        p = " ".join(p.split())
        if total + len(p) <= limit_chars:
            combined.append(p)
            total += len(p)
        else:
            break
    return "\n".join(f"- {textwrap.shorten(p, width=400, placeholder=' ...')}" for p in combined)

# =========================
# Embedding-based similarity
# =========================
class SimilarityGuard:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def most_similar(self, text: str, corpus: List[str]) -> float:
        if not corpus:
            return 0.0
        v_new = self.model.encode([text], normalize_embeddings=True)
        v_past = self.model.encode(corpus, normalize_embeddings=True)
        sims = util.cos_sim(v_new, v_past)[0].cpu().tolist()
        return max(sims)

# =========================
# Local LLM wrapper
# =========================
class LocalChatModel:
    def __init__(self, model_id: str, device: str, dtype):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.8, top_p: float = 0.9,
             max_new_tokens: int = 512) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in text:
            text = text[len(prompt):]
        return text.strip()

# =========================
# Sanitizers
# =========================
META_PATTERNS = (
    r"^\s*system\b.*$",
    r"^\s*user\b.*$",
    r"^\s*assistant\b.*$",
    r"^\s*Cutting Knowledge Date:.*$",
    r"^\s*Today Date:.*$",
    r"^\s*You write YouTube Shorts narration scripts.*$",
    r"^\s*Use ONLY the provided facts.*$",
    r"^\s*Start with a 1-sentence hook.*$",
    r"^\s*Keep sentences short for TTS.*$",
    r"^\s*End with a 1-sentence punch.*$",
    r"^\s*Write only the narration text.*$",
    r"^\s*Topic:.*$",
    r"^\s*Facts \(from Wikipedia\):.*$",
    r"^\s*Constraints:.*$",
)

def strip_meta_lines(text: str) -> str:
    lines = [ln for ln in text.splitlines()]
    clean = []
    for ln in lines:
        if any(re.search(p, ln, flags=re.IGNORECASE) for p in META_PATTERNS):
            continue
        clean.append(ln)
    return "\n".join([l for l in clean if l.strip()])

def sanitize_narration(text: str) -> str:
    text = strip_meta_lines(text)
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not parts:
        return text.strip()
    best = max(parts, key=len)
    return best.strip()

_NUM_LINE = re.compile(r"^\s*(\d{1,2})\.\s+(.*\S)\s*$")

def parse_numbered_list(text: str) -> List[str]:
    text = strip_meta_lines(text)
    prompts = []
    found_numbered = False
    for ln in text.splitlines():
        m = _NUM_LINE.match(ln)
        if m:
            found_numbered = True
            idx, content = int(m.group(1)), m.group(2).strip()
            if 1 <= idx <= 10 and content:
                prompts.append((idx, content))
        elif found_numbered:
            # Stop at the first non-numbered line after the list starts
            break
    if prompts:
        prompts.sort(key=lambda x: x[0])
        return [c for _, c in prompts][:10]
    fallback = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return fallback[:10]

# =========================
# Generation logic
# =========================
def word_count(text: str) -> int:
    return len(text.split())

def enforce_word_range(text: str, min_words: int, max_words: int) -> str:
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text

def make_user_msg(topic: str, fact_block: str) -> str:
    return (
        f"Topic: {topic}\n"
        f"Facts (from Wikipedia):\n{fact_block}\n"
        f"Constraints: 300–500 words."
    )

def generate_script(topic: str, fact_block: str, llm: LocalChatModel) -> str:
    user_msg = make_user_msg(topic, fact_block)
    draft = llm.chat(GEN_PROMPT, user_msg, temperature=TEMPERATURE, top_p=TOP_P, max_new_tokens=MAX_NEW_TOKENS)
    return sanitize_narration(draft)

def rewrite_script(original: str, fact_block: str, llm: LocalChatModel) -> str:
    user_msg = f"Facts:\n{fact_block}\n---\nOriginal script:\n{original}"
    rewrite = llm.chat(REWRITE_PROMPT, user_msg, temperature=TEMPERATURE, top_p=TOP_P, max_new_tokens=MAX_NEW_TOKENS)
    return sanitize_narration(rewrite)

def generate_image_prompts(topic: str, llm: LocalChatModel) -> List[str]:
    user_msg = f"Topic: {topic}"
    raw = llm.chat(IMAGE_PROMPT_SYSTEM, user_msg, temperature=0.7, top_p=0.9, max_new_tokens=256)
    prompts = parse_numbered_list(raw)
    out = []
    for p in prompts:
        out.append(p if topic.lower() in p.lower() else f"{topic}: {p}")
    while len(out) < 10:
        out.append(f"{topic}: cinematic wide shot, detailed, high resolution")
    return out[:10]

# =========================
# Variable format output
# =========================
def escape_for_python_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\"", "\\\"")

def format_story_variables(idx: int, narration: str, prompts: List[str]) -> str:
    narr = escape_for_python_string(narration)
    lines = [f'story_{idx}="{narr}"', f"pic_prompts_{idx} = ["]
    for i, p in enumerate(prompts):
        sep = "," if i < len(prompts)-1 else ""
        lines.append(f'    "{escape_for_python_string(p)}"{sep}')
    lines.append("]")
    return "\n".join(lines)

def format_image_prompt_list(var_name: str, prompts: List[str]) -> str:
    lines = [f"{var_name} = ["]
    for i, p in enumerate(prompts):
        sep = "," if i < len(prompts) - 1 else ""
        lines.append(f'    "{escape_for_python_string(p)}"{sep}')
    lines.append("]")
    return "\n".join(lines)

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()
    con = connect_db()
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="yt-short-rag/2.0 (contact: youremail@example.com)"
    )

    topic = pick_topic(wiki, con)
    passages = wiki_passages(wiki, topic, max_passages=5)
    if not passages:
        print("No Wikipedia passages found. Try again.")
        return
    fact_block = build_fact_block(passages)

    guard = SimilarityGuard(EMB_MODEL)
    llm = LocalChatModel(MODEL_ID, DEVICE, DTYPE)

    draft = generate_script(topic, fact_block, llm)
    draft = enforce_word_range(draft, MIN_WORDS, MAX_WORDS)

    attempts = 0
    while word_count(draft) < MIN_WORDS and attempts < 2:
        draft = enforce_word_range(rewrite_script(draft, fact_block, llm), MIN_WORDS, MAX_WORDS)
        attempts += 1

    past = get_recent_texts(con, PAST_K)
    sim = guard.most_similar(draft, past)
    if sim >= SIM_THRESHOLD:
        draft = enforce_word_range(rewrite_script(draft, fact_block, llm), MIN_WORDS, MAX_WORDS)

    img_prompts = generate_image_prompts(topic, llm)

    save_script(con, topic, draft)
    narr_path, img_path = save_to_file(topic, draft, img_prompts)

    print("\n=== TOPIC ===")
    print(topic)
    print("\n=== NARRATION (300–500 words) ===")
    print(draft)
    print("\n=== IMAGE PROMPTS ===")
    for i, p in enumerate(img_prompts, 1):
        print(f"{i}. {p}")

    print("\n=== VARIABLE FORMAT OUTPUT ===")
    print(format_story_variables(1, draft, img_prompts))

    print(f"\nSaved narration to: {narr_path}")
    print(f"Saved image prompts to: {img_path}")

# def main():
#     ensure_dirs()
#     con = connect_db()
#     recent_texts = get_recent_texts(con, PAST_K)
#     print(recent_texts[0])
    
if __name__ == "__main__":
    main()