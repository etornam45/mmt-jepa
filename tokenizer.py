"""
tokenizer.py — train a joint English+Twi BPE tokenizer with SentencePiece.

Pulls text from the same HuggingFace dataset used by ObjB, writes a
plain .txt corpus, trains the model, and saves tokenizer.model + tokenizer.vocab.

Run:
    python tokenizer.py
"""

from datasets import load_dataset
import sentencepiece as spm

VOCAB_SIZE = 16_000  # good for 2-language system
MODEL_PREFIX = "model/tokenizer"
CORPUS_FILE = "corpus.txt"

print("Loading dataset...")
ds = load_dataset(
    "ghananlpcommunity/twi-english-paragraph-dataset_news",
    split="train",
)

print(f"Writing corpus ({len(ds):,} rows × 2 languages)...")
with open(CORPUS_FILE, "w", encoding="utf-8") as f:
    for row in ds:
        eng = str(row.get("ENGLISH") or "").strip()
        twi = str(row.get("TWI") or "").strip()
        if eng:
            f.write(eng + "\n")
        if twi:
            f.write(twi + "\n")

n_lines = sum(1 for _ in open(CORPUS_FILE, encoding="utf-8"))
print(f"Corpus: {n_lines:,} lines  →  {CORPUS_FILE}")

print(f"\nTraining BPE tokenizer  vocab_size={VOCAB_SIZE} ...")
spm.SentencePieceTrainer.Train(
    input=CORPUS_FILE,
    model_prefix=MODEL_PREFIX,
    model_type="bpe",
    vocab_size=VOCAB_SIZE,
    character_coverage=0.9999,  # high coverage for Twi special chars (ɛ ɔ ɑ)
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
    num_threads=4,
)

# ── 3. verify ─────────────────────────────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.Load(f"{MODEL_PREFIX}.model")

eng_test = "Ghana's Black Queens face a must-win clash against Mali."
twi_test = "Ɔmanpanyin John Dramani Mahama de ne nnɔnhwerew aduonu anan sikasɛm."

print(f"\nVocab size : {sp.GetPieceSize()}")
print(f"\nEN → {sp.Encode(eng_test)[:12]}...")
print(f"TW → {sp.Encode(twi_test)[:12]}...")
print(f"\nEN decoded → {sp.Decode(sp.Encode(eng_test))}")
print(f"TW decoded → {sp.Decode(sp.Encode(twi_test))}")
print(f"\nSaved: {MODEL_PREFIX}.model  {MODEL_PREFIX}.vocab")
