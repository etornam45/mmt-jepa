import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import ModelConfig
from dataset import ObjA, ObjB, ObjC
from decoder import Decoder, mel_reconstruction_loss
from logger import TrainingLogger
from model import MMT_JEPA

EPOCHS      = 10
LR          = 3e-4
BATCH_SIZE  = 32
LOG_EVERY   = 200
GRAD_CLIP   = 1.0
FREEZE_JEPA = True
JEPA_CKPT   = "checkpoints/epoch010.pt"
OUT_DIR     = "checkpoints"
MAX_SAMPLES = None

# Toggle objectives — any non-empty subset of {"A", "B", "C"}
#   "A" : audio  -> text   (transcription)
#   "B" : text   -> text   (translation)
#   "C" : text   -> audio  (TTS)
ACTIVE = {"B"}

_OBJ_ORDER = ["B"]   # canonical rotation order

assert ACTIVE and ACTIVE <= {"A", "B", "C"}, \
    f"ACTIVE must be a non-empty subset of {{'A','B','C'}}, got {ACTIVE!r}"


PAD_ID = 0   # must match dataset.PAD


def text_ce_loss(logits: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
    """
    Autoregressive cross-entropy with teacher forcing.

    logits  : (B, L-1, vocab_size) — model predictions for positions 1..L
    tgt_ids : (B, L)               — ground-truth token ids (unshifted)

    The target is tgt_ids[:, 1:] so position i predicts token i+1.
    Padding positions (PAD_ID) are masked out of the loss.
    """
    # logits: (B, L-1, V)  →  targets: tgt_ids[:, 1:] shape (B, L-1)
    targets = tgt_ids[:, 1:].contiguous()                          # (B, L-1)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),                       # (B*(L-1), V)
        targets.reshape(-1),                                       # (B*(L-1),)
        ignore_index=PAD_ID,
    )


def forward_obj(
    obj: str,
    b: dict,
    model: Decoder,
    *,
    freeze_jepa: bool,
) -> torch.Tensor:
    """Run one objective and return its loss.

    When ``freeze_jepa`` is True, encoder + predictor (and frozen ``text_stem`` for
    teacher-forcing inputs) run under ``torch.no_grad()`` so activations are not
    saved for backprop through unused frozen weights.
    """
    enc_cm = torch.no_grad() if freeze_jepa else nullcontext()

    if obj == "A":
        # ObjA: audio -> text  (teacher-forced CE)
        with enc_cm:
            z_ctx = model.encode(audio_mel=b["ctx_audio"], pad_mask=b["ctx_pad_mask"])
            z_hat = model.predict(z_ctx, b["src_lang"], b["src_mod"], b["tgt_lang"], b["tgt_mod"])
            tgt_in = model.text_stem(b["tgt_text"][:, :-1])
        logits = model.text_decoder(tgt_in, z_hat, b["tgt_mod"], b["tgt_lang"])
        return text_ce_loss(logits, b["tgt_text"])

    if obj == "B":
        # ObjB: text -> text  (teacher-forced CE)
        with enc_cm:
            z_ctx = model.encode(text_ids=b["ctx_text"], pad_mask=b["ctx_pad_mask"])
            z_hat = model.predict(z_ctx, b["src_lang"], b["src_mod"], b["tgt_lang"], b["tgt_mod"])
            tgt_in = model.text_stem(b["tgt_text"][:, :-1])
        logits = model.text_decoder(tgt_in, z_hat, b["tgt_mod"], b["tgt_lang"])
        return text_ce_loss(logits, b["tgt_text"])

    if obj == "C":
        # ObjC: text -> audio
        with enc_cm:
            z_ctx = model.encode(text_ids=b["ctx_text"], pad_mask=b["ctx_pad_mask"])
            z_hat = model.predict(z_ctx, b["src_lang"], b["src_mod"], b["tgt_lang"], b["tgt_mod"])
        pred = model.audio_decoder(z_hat, b["tgt_lang"], target_len=b["tgt_audio"].size(-1))
        return mel_reconstruction_loss(pred, b["tgt_audio"])

    raise ValueError(f"Unknown objective: {obj!r}")


def decoder_checkpoint_state_dict(model: Decoder) -> dict[str, torch.Tensor]:
    """Trainable heads only — avoids a full copy of frozen JEPA weights each save."""
    out: dict[str, torch.Tensor] = {}
    out.update({f"text_decoder.{k}": v for k, v in model.text_decoder.state_dict().items()})
    out.update({f"audio_decoder.{k}": v for k, v in model.audio_decoder.state_dict().items()})
    return out


def maybe_load_jepa_checkpoint(model: MMT_JEPA, ckpt_path: str | None) -> None:
    """Load weights with ``strict=False`` (LeJEPA ckpts omit ``target_*``; legacy EMA ckpts drop targets)."""
    if not ckpt_path:
        return
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint : {ckpt_path}")
    print(f"  missing keys    : {len(missing)}")
    print(f"  unexpected keys : {len(unexpected)}")


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device:", device)
    print(f"Active objectives : {sorted(ACTIVE)}")
    run_log = TrainingLogger.create(name="decoder")
    print(f"Metrics dir: {run_log.run_path}")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("tokenizer/tokenizer.model")

    cfg = ModelConfig()

    _dataset_cls = {"A": ObjA, "B": ObjB, "C": ObjC}
    active_objs  = [o for o in _OBJ_ORDER if o in ACTIVE]   # stable order
    loaders      = {
        o: _dataset_cls[o](sp, cfg, max_samples=MAX_SAMPLES).loader(batch_size=BATCH_SIZE, num_workers=2)
        for o in active_objs
    }

    steps_per_epoch = sum(len(loader) for loader in loaders.values())
    total_steps     = max(1, steps_per_epoch * EPOCHS)
    print(f"Steps/epoch: {steps_per_epoch:,}  total: {total_steps:,}")

    base  = MMT_JEPA(cfg)
    maybe_load_jepa_checkpoint(base, JEPA_CKPT)
    print(f"Jepa Parameters: {sum(p.numel() for p in base.parameters())}")
    model = Decoder(cfg, base, freeze_jepa=FREEZE_JEPA).to(device)
    print(f"Decoder Parameters: {sum(p.numel() for p in model.parameters())}")
    del base
    # Load checkpont
    # model.load_state_dict(torch.load("checkpoints/jepa_ep003_decoder_epoch010.pt"))
    last_epoch = 0

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters : {trainable:,}")
    print(f"Total parameters     : {total:,}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01, betas=(0.9, 0.95),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: (
        s / max(1, 500) if s < 500
        else max(0.05, 0.5 * (1 + math.cos(math.pi * (s - 500) / max(1, total_steps - 500))))
    ))

    os.makedirs(OUT_DIR, exist_ok=True)
    step = 0
    model.train()

    for epoch in range(last_epoch, last_epoch + EPOCHS):
        t0      = time.time()
        running = 0.0
        iters   = {o: iter(loader) for o, loader in loaders.items()}

        for idx in tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{EPOCHS}"):
            obj = active_objs[idx % len(active_objs)]

            try:
                batch = next(iters[obj])
            except StopIteration:
                iters[obj] = iter(loaders[obj])
                try:
                    batch = next(iters[obj])
                except StopIteration:
                    continue
            if not batch:
                continue

            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

            opt.zero_grad(set_to_none=True)
            loss = forward_obj(obj, b, model, freeze_jepa=FREEZE_JEPA)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
            )
            opt.step()
            sched.step()

            running += loss.item()
            step    += 1

            if step % LOG_EVERY == 0:
                print(
                    f"\nstep {step:06d} obj={obj}  "
                    f"loss {loss.item():.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}"
                )
                run_log.log_step(
                    {
                        "step": step,
                        "epoch": epoch + 1,
                        "obj": obj,
                        "loss": loss.item(),
                        "lr": sched.get_last_lr()[0],
                    }
                )

        avg = running / max(1, steps_per_epoch)
        print(f"epoch {epoch + 1}/{EPOCHS}  avg_loss {avg:.4f}  {time.time() - t0:.0f}s")
        run_log.log_epoch(epoch + 1, avg, time.time() - t0)

        ckpt_path = os.path.join(OUT_DIR, f"jepa_ep003_decoder_epoch{epoch + 1:03d}.pt")
        torch.save(
            decoder_checkpoint_state_dict(model) if FREEZE_JEPA else model.state_dict(),
            ckpt_path,
        )
        print(f"Saved: {ckpt_path}")

    run_log.close()