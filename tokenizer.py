"""
tokenizer.py — train a joint English+Twi BPE tokenizer with SentencePiece.

Pulls text from the same HuggingFace dataset used by ObjB, writes a
plain .txt corpus, trains the model, and saves tokenizer.model + tokenizer.vocab.

Run:
    python tokenizer.py
"""

from datasets import load_dataset
import sentencepiece as spm

VOCAB_SIZE   = 16_000 
MODEL_PREFIX = "model/tokenizer"
CORPUS_FILE  = "corpus.txt"

SOURCES = [
    ("ghananlpcommunity/twi-english-paragraph-dataset_news", "ENGLISH", "TWI"),
    ("ghananlpcommunity/english-twi-sentences-non-nouns",    "english", "twi"),
    ("ghananlpcommunity/english-twi-nouns-v2",               "english", "twi"),
]

print(f"Writing corpus to {CORPUS_FILE} ...")
with open(CORPUS_FILE, "w", encoding="utf-8") as f:
    for repo, eng_col, twi_col in SOURCES:
        try:
            ds = load_dataset(repo, split="train")
            for row in ds:
                eng = str(row.get(eng_col) or "").strip()
                twi = str(row.get(twi_col) or "").strip()
                if eng: 
                    f.write(eng + "\n")
                if twi: 
                    f.write(twi + "\n")
            print(f"  {repo.split('/')[1]}: {len(ds):,} rows")
        except Exception as e:
            print(f"  {repo.split('/')[1]} skipped ({e})")

n_lines = sum(1 for _ in open(CORPUS_FILE, encoding="utf-8"))
print(f"Corpus: {n_lines:,} lines  →  {CORPUS_FILE}")

print(f"\nTraining BPE tokenizer  vocab_size={VOCAB_SIZE} ...")
spm.SentencePieceTrainer.Train(
    input            = CORPUS_FILE,
    model_prefix     = MODEL_PREFIX,
    model_type       = "bpe",
    vocab_size       = VOCAB_SIZE,
    character_coverage = 0.9999,    # high coverage for Twi special chars (ɛ ɔ ɑ)
    pad_id           = 0,
    unk_id           = 1,
    bos_id           = 2,
    eos_id           = 3,
    pad_piece        = "<pad>",
    unk_piece        = "<unk>",
    bos_piece        = "<s>",
    eos_piece        = "</s>",
    num_threads      = 4,
)

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