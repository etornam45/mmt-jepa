"""
MMT-JEPA datasets — one class per JEPA objective.

    ObjA  Audio -> Text   (both languages)
          English: openslr/librispeech_asr  train.100
          Twi:     ghananlpcommunity/twi-speech-text-multispeaker-16k
                   + BibleTTS local (openslr.org/129)

    ObjB  Text -> Text    (both languages, translation)
          ghananlpcommunity/twi-english-paragraph-dataset_news

    ObjC  Text -> Audio   (both languages)
          Same audio sources as ObjA, direction flipped

Quick start:
    tok = MyTokenizer()          # any object with .encode(str) -> list[int]
    for batch in ObjA(tok).loader(): ...
    for batch in ObjB(tok).loader(): ...
    for batch in ObjC(tok).loader(): ...
"""

from __future__ import annotations
import io
import random
import numpy as np
import soundfile as sf
import torch
import librosa
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ENG, TWI    = 0, 1
TEXT, AUDIO = 0, 1
PAD = 0


def make_mel(audio, src_sr: int, cfg=None,
             target_sr: int = 16_000, n_mels: int = 80) -> Tensor | None:
    """
    audio: numpy array or torch Tensor, any shape — mono or stereo.
    Returns (n_mels, T) float32 tensor, z-score normalised. None if too long/short.
    Accepts an optional ModelConfig; if provided, uses cfg.sample_rate and cfg.n_mels.
    """
    if cfg is not None:
        target_sr = cfg.sample_rate
        n_mels    = cfg.n_mels

    if isinstance(audio, Tensor):
        y = audio.numpy()
    else:
        y = np.asarray(audio, dtype=np.float32)

    if y.ndim > 1:
        y = y.mean(axis=0)

    if src_sr != target_sr:
        y = librosa.resample(y, orig_sr=src_sr, target_sr=target_sr)

    if len(y) == 0 or len(y) > target_sr * 30:
        return None

    mel    = librosa.feature.melspectrogram(
        y=y, sr=target_sr, n_fft=400, hop_length=160, n_mels=n_mels, power=2.0
    )
    mel_db = librosa.power_to_db(mel, top_db=80)
    mel_t  = torch.from_numpy(mel_db.astype(np.float32))
    return (mel_t - mel_t.mean()) / (mel_t.std() + 1e-6)


class _LibriSpeech(Dataset):
    """English audio + transcript from openslr/librispeech_asr."""
    def __init__(self, tokenizer, cfg=None, max_text: int = 256, split: str = "train.100") -> None:
        from datasets import Audio, load_dataset
        ds = load_dataset("openslr/librispeech_asr", "clean", split=split)
        self.ds      = ds.cast_column("audio", Audio(decode=False))
        self.tok     = tokenizer
        self.cfg     = cfg
        self.max_txt = cfg.max_seq_len if cfg is not None else max_text
        print(f"  LibriSpeech ({split}): {len(self.ds):,} utterances")

    def __len__(self):  return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        try:
            info = row["audio"]
            raw  = info.get("bytes") or open(info["path"], "rb").read()
            audio, sr = sf.read(io.BytesIO(raw))
            audio = audio.astype(np.float32)
        except Exception:
            return None
        mel = make_mel(audio, sr, self.cfg)
        if mel is None:
            return None
        ids = self.tok.encode(row["text"])[:self.max_txt]
        return {"mel": mel, "ids": torch.tensor(ids, dtype=torch.long), "lang": ENG}


class _TwiAudio(Dataset):
    """
    Twi audio + transcript.
    Sources:
      1. BibleTTS local  (set bibletts_dir, expects train/<stem>.flac + .txt)
      2. ghananlpcommunity/twi-speech-text-multispeaker-16k  (HuggingFace)
    """
    def __init__(self, tokenizer, cfg=None, split: str = "train", max_text: int = 256) -> None:
        self.tok     = tokenizer
        self.cfg     = cfg
        self.max_txt = cfg.max_seq_len if cfg is not None else max_text
        self.records: list[dict] = []

        try:
            from datasets import Audio, load_dataset
            # decode=False avoids torchcodec; we decode bytes with soundfile
            ds = load_dataset("ghananlpcommunity/twi-speech-text-multispeaker-16k", split=split)
            ds = ds.cast_column("audio", Audio(decode=False))
            before = len(self.records)
            for row in ds:
                text = (row.get("text") or row.get("sentence") or row.get("transcription") or "").strip()
                if not text:
                    continue
                info = row["audio"]
                raw  = info.get("bytes") or open(info["path"], "rb").read()
                self.records.append({"bytes": raw, "text": text})
            print(f"  Twi HuggingFace ({split}): +{len(self.records) - before:,} utterances")
        except Exception as e:
            print(f"  Twi HuggingFace skipped ({e})")

        assert self.records, (
            "No Twi audio found.\n"
            "  Set bibletts_dir to your BibleTTS extract or ensure internet access."
        )

    def __len__(self):  return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        try:
            audio, sr = sf.read(io.BytesIO(rec["bytes"]))
            audio = audio.astype(np.float32)
        except Exception:
            return None
        mel = make_mel(audio, sr, self.cfg)
        if mel is None:
            return None
        ids = self.tok.encode(rec["text"])[:self.max_txt]
        if not ids:
            return None
        return {"mel": mel, "ids": torch.tensor(ids, dtype=torch.long), "lang": TWI}


class ObjA(Dataset):
    """
    Context : audio  (English or Twi)
    Target  : text   (same language)

    Batch keys
    ----------
    ctx_audio (B, 80, T) | ctx_text None
    tgt_text  (B, L)     | tgt_audio None
    ctx_pad_mask (B, T)  | tgt_pad_mask (B, L)
    src_lang / tgt_lang / src_mod (=AUDIO) / tgt_mod (=TEXT)
    """

    def __init__(self, tokenizer, cfg=None, max_text: int = 256) -> None:
        print("ObjA sources:")
        eng = _LibriSpeech(tokenizer, cfg, max_text)
        twi = _TwiAudio(tokenizer, cfg, max_text=max_text)
        self.ds = ConcatDataset([eng, twi])

    def __len__(self):  return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        if item is None:
            return None
        return {
            "mel":     item["mel"],
            "ctx_ids": item["ids"],   # audio is context
            "tgt_ids": None,
            "src_lang": item["lang"], "tgt_lang": item["lang"],
            "src_mod": AUDIO, "tgt_mod": TEXT,
        }

    def loader(self, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=True, num_workers=num_workers,
                          collate_fn=_collate_audio_text, drop_last=True)


class ObjB(Dataset):
    """
    Context : text  (English or Twi)
    Target  : text  (other language)
    Direction randomly flipped each call so model sees both eng->twi and twi->eng.

    Batch keys
    ----------
    ctx_audio None | ctx_text (B, L)
    tgt_audio None | tgt_text (B, L)
    ctx_pad_mask (B, L) | tgt_pad_mask (B, L)
    src_lang / tgt_lang / src_mod (=TEXT) / tgt_mod (=TEXT)
    """

    SOURCES = [
        ("ghananlpcommunity/twi-english-paragraph-dataset_news", {}, "ENGLISH", "TWI"),
        ("ghananlpcommunity/english-twi-sentences-non-nouns",    {}, "english", "twi"),
        ("ghananlpcommunity/english-twi-nouns-v2",               {}, "english", "twi"),
        # add more (repo, kwargs, eng_col, twi_col) here as datasets become available
    ]

    def __init__(self, tokenizer, cfg=None, max_text: int = 256, reverse_prob: float = 0.5) -> None:
        from datasets import load_dataset
        self.tok          = tokenizer
        self.max_txt      = cfg.max_seq_len if cfg is not None else max_text
        self.reverse_prob = reverse_prob
        self.pairs: list[tuple[str, str]] = []

        print("ObjB sources:")
        for repo, kwargs, eng_col, twi_col in self.SOURCES:
            try:
                ds = load_dataset(repo, split="train", **kwargs)
                before = len(self.pairs)
                for row in ds:
                    eng = str(row.get(eng_col) or "").strip()
                    twi = str(row.get(twi_col) or "").strip()
                    if eng and twi:
                        self.pairs.append((eng, twi))
                print(f"  {repo.split('/')[1]}: +{len(self.pairs) - before:,} pairs")
            except Exception as e:
                print(f"  {repo.split('/')[1]} skipped ({e})")

        assert self.pairs, "ObjB: no data loaded — check internet or sources list"

    def __len__(self):  return len(self.pairs)

    def __getitem__(self, i):
        eng, twi = self.pairs[i]
        if random.random() < self.reverse_prob:
            ctx, tgt, sl, tl = twi, eng, TWI, ENG
        else:
            ctx, tgt, sl, tl = eng, twi, ENG, TWI
        return {
            "ctx_ids": torch.tensor(self.tok.encode(ctx)[:self.max_txt], dtype=torch.long),
            "tgt_ids": torch.tensor(self.tok.encode(tgt)[:self.max_txt], dtype=torch.long),
            "src_lang": sl, "tgt_lang": tl,
            "src_mod": TEXT, "tgt_mod": TEXT,
        }

    def loader(self, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=True, num_workers=num_workers,
                          collate_fn=_collate_text_text, drop_last=True)


class ObjC(Dataset):
    """
    Context : text   (English or Twi)
    Target  : audio  (same language)
    Same underlying data as ObjA, direction flipped.

    Batch keys
    ----------
    ctx_audio None  | ctx_text (B, L)
    tgt_audio (B, 80, T) | tgt_text None
    ctx_pad_mask (B, L)  | tgt_pad_mask (B, T)
    src_lang / tgt_lang / src_mod (=TEXT) / tgt_mod (=AUDIO)
    """

    def __init__(self, tokenizer, cfg=None, max_text: int = 256) -> None:
        print("ObjC sources:")
        eng = _LibriSpeech(tokenizer, cfg, max_text)
        twi = _TwiAudio(tokenizer, cfg, max_text=max_text)
        self.ds = ConcatDataset([eng, twi])

    def __len__(self):  return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        if item is None:
            return None
        return {
            "mel":     item["mel"],
            "ctx_ids": item["ids"],   # text is context, same ids used for lookup
            "tgt_ids": None,
            "src_lang": item["lang"], "tgt_lang": item["lang"],
            "src_mod": TEXT, "tgt_mod": AUDIO,
        }

    def loader(self, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=True, num_workers=num_workers,
                          collate_fn=_collate_text_audio, drop_last=True)


def collate_fn(batch):
    """
    Unified collate for mixed ObjA + ObjB + ObjC batches.
    All items share the same keys:
        mel      : (n_mels, T) or None
        ctx_ids  : (L,) token ids — context text/audio tokens
        tgt_ids  : (L,) token ids — target text tokens (None for audio target)
        src_mod / tgt_mod / src_lang / tgt_lang
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    src_mod = torch.tensor([b["src_mod"] for b in batch])
    tgt_mod = torch.tensor([b["tgt_mod"] for b in batch])
    src_lang = torch.tensor([b["src_lang"] for b in batch])
    tgt_lang = torch.tensor([b["tgt_lang"] for b in batch])

    # ── audio (present when src_mod=AUDIO or tgt_mod=AUDIO) ──────────────────
    has_mel = any(b.get("mel") is not None for b in batch)
    if has_mel:
        mels   = [b["mel"] for b in batch if b.get("mel") is not None]
        max_T  = max(m.shape[1] for m in mels)
        n_mels = mels[0].shape[0]
        audio_pad  = torch.zeros(len(batch), n_mels, max_T)
        audio_mask = torch.ones(len(batch), max_T, dtype=torch.bool)
        mel_idx = 0
        for i, b in enumerate(batch):
            if b.get("mel") is not None:
                t = b["mel"].shape[1]
                audio_pad[i, :, :t] = b["mel"]
                audio_mask[i, :t]   = False
                mel_idx += 1
    else:
        audio_pad = audio_mask = None

    # ── context text (present when src_mod=TEXT) ──────────────────────────────
    has_ctx_text = any(b["ctx_ids"] is not None and b["src_mod"] == TEXT for b in batch)
    if has_ctx_text:
        ctx_seqs  = [b["ctx_ids"] if (b["ctx_ids"] is not None and b["src_mod"] == TEXT)
                     else torch.zeros(1, dtype=torch.long) for b in batch]
        ctx_text  = pad_sequence(ctx_seqs, batch_first=True, padding_value=PAD)
        ctx_tmask = ctx_text == PAD
    else:
        ctx_text = ctx_tmask = None

    # ── target text (present when tgt_mod=TEXT) ───────────────────────────────
    has_tgt_text = any(b["tgt_ids"] is not None and b["tgt_mod"] == TEXT for b in batch)
    if has_tgt_text:
        tgt_seqs  = [b["tgt_ids"] if (b["tgt_ids"] is not None and b["tgt_mod"] == TEXT)
                     else torch.zeros(1, dtype=torch.long) for b in batch]
        tgt_text  = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD)
        tgt_tmask = tgt_text == PAD
    else:
        tgt_text = tgt_tmask = None

    # route audio to ctx or tgt based on modality
    is_audio_ctx = src_mod[0].item() == AUDIO if has_mel else False
    ctx_audio    = audio_pad  if (has_mel and is_audio_ctx)  else None
    tgt_audio    = audio_pad  if (has_mel and not is_audio_ctx) else None
    ctx_amask    = audio_mask if (has_mel and is_audio_ctx)  else None
    tgt_amask    = audio_mask if (has_mel and not is_audio_ctx) else None

    return {
        "ctx_audio":    ctx_audio,
        "ctx_text":     ctx_text,
        "tgt_audio":    tgt_audio,
        "tgt_text":     tgt_text,
        "ctx_pad_mask": ctx_amask if ctx_audio is not None else ctx_tmask,
        "tgt_pad_mask": tgt_amask if tgt_audio is not None else tgt_tmask,
        "src_lang": src_lang, "tgt_lang": tgt_lang,
        "src_mod":  src_mod,  "tgt_mod":  tgt_mod,
    }


def _collate_audio_text(batch):
    """Audio context, text target  (ObjA)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    max_T  = max(b["mel"].shape[1] for b in batch)
    n_mels = batch[0]["mel"].shape[0]
    ctx_audio = torch.zeros(len(batch), n_mels, max_T)
    ctx_mask  = torch.ones(len(batch), max_T, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["mel"].shape[1]
        ctx_audio[i, :, :t] = b["mel"]
        ctx_mask[i, :t]     = False
    tgt_text = pad_sequence([b["ids"] for b in batch], batch_first=True, padding_value=PAD)
    return {
        "ctx_audio": ctx_audio, "ctx_text": None,
        "tgt_audio": None,      "tgt_text": tgt_text,
        "ctx_pad_mask": ctx_mask, "tgt_pad_mask": tgt_text == PAD,
        "src_lang": torch.tensor([b["src_lang"] for b in batch]),
        "tgt_lang": torch.tensor([b["tgt_lang"] for b in batch]),
        "src_mod":  torch.tensor([b["src_mod"]  for b in batch]),
        "tgt_mod":  torch.tensor([b["tgt_mod"]  for b in batch]),
    }


def _collate_text_text(batch):
    """Text context, text target  (ObjB)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    ctx = pad_sequence([b["ctx_ids"] for b in batch], batch_first=True, padding_value=PAD)
    tgt = pad_sequence([b["tgt_ids"] for b in batch], batch_first=True, padding_value=PAD)
    return {
        "ctx_audio": None, "ctx_text": ctx,
        "tgt_audio": None, "tgt_text": tgt,
        "ctx_pad_mask": ctx == PAD, "tgt_pad_mask": tgt == PAD,
        "src_lang": torch.tensor([b["src_lang"] for b in batch]),
        "tgt_lang": torch.tensor([b["tgt_lang"] for b in batch]),
        "src_mod":  torch.tensor([b["src_mod"]  for b in batch]),
        "tgt_mod":  torch.tensor([b["tgt_mod"]  for b in batch]),
    }


def _collate_text_audio(batch):
    """Text context, audio target  (ObjC)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    ctx_text  = pad_sequence([b["ids"] for b in batch], batch_first=True, padding_value=PAD)
    max_T     = max(b["mel"].shape[1] for b in batch)
    n_mels    = batch[0]["mel"].shape[0]
    tgt_audio = torch.zeros(len(batch), n_mels, max_T)
    tgt_mask  = torch.ones(len(batch), max_T, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["mel"].shape[1]
        tgt_audio[i, :, :t] = b["mel"]
        tgt_mask[i, :t]     = False
    return {
        "ctx_audio": None,      "ctx_text": ctx_text,
        "tgt_audio": tgt_audio, "tgt_text": None,
        "ctx_pad_mask": ctx_text == PAD, "tgt_pad_mask": tgt_mask,
        "src_lang": torch.tensor([b["src_lang"] for b in batch]),
        "tgt_lang": torch.tensor([b["tgt_lang"] for b in batch]),
        "src_mod":  torch.tensor([b["src_mod"]  for b in batch]),
        "tgt_mod":  torch.tensor([b["tgt_mod"]  for b in batch]),
    }


if __name__ == "__main__":

    class _Tok:
        def encode(self, text: str) -> list[int]:
            return [ord(c) % 20_000 for c in text[:64]]

    tok = _Tok()

    def _grab(ds, n=4):
        items = []
        for i in range(min(len(ds), 200)):
            x = ds[i]
            if x is not None:
                items.append(x)
            if len(items) == n:
                break
        return items

    print("\n── Obj A: Audio -> Text (both languages) ──")
    try:
        ds = ObjA(tok)
        b  = _collate_audio_text(_grab(ds))
        print(f"  ctx_audio    : {b['ctx_audio'].shape}")
        print(f"  tgt_text     : {b['tgt_text'].shape}")
        print(f"  src_mod      : {b['src_mod'].tolist()}  (1=audio)")
        print(f"  tgt_mod      : {b['tgt_mod'].tolist()}  (0=text)")
        print(f"  src_lang mix : {b['src_lang'].tolist()}  (0=eng 1=twi)")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL — {e}")

    print("\n── Obj B: Text -> Text (translation) ──")
    try:
        ds = ObjB(tok)
        b  = _collate_text_text(_grab(ds))
        print(f"  ctx_text     : {b['ctx_text'].shape}")
        print(f"  tgt_text     : {b['tgt_text'].shape}")
        print(f"  src_mod      : {b['src_mod'].tolist()}  (0=text)")
        print(f"  src_lang mix : {b['src_lang'].tolist()}  (0=eng 1=twi)")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL — {e}")

    print("\n── Obj C: Text -> Audio (both languages) ──")
    try:
        ds = ObjC(tok)
        b  = _collate_text_audio(_grab(ds))
        print(f"  ctx_text     : {b['ctx_text'].shape}")
        print(f"  tgt_audio    : {b['tgt_audio'].shape}")
        print(f"  src_mod      : {b['src_mod'].tolist()}  (0=text)")
        print(f"  tgt_mod      : {b['tgt_mod'].tolist()}  (1=audio)")
        print(f"  src_lang mix : {b['src_lang'].tolist()}  (0=eng 1=twi)")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL — {e}")