import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import ObjA, ObjC
from model import MMT_JEPA, ModelConfig
from decoder import Decoder, mel_reconstruction_loss


EPOCHS = 10
LR = 3e-4
BATCH_SIZE = 32
LOG_EVERY = 200
GRAD_CLIP = 1.0


def text_reconstruction_loss(
    pred_text_emb: torch.Tensor,
    tgt_text_ids: torch.Tensor,
    text_stem: torch.nn.Module,
    l1_weight: float = 1.0,
    mse_weight: float = 0.5,
) -> torch.Tensor:
    """
    Train text decoder to reconstruct target token embeddings.
    Uses frozen text stem embeddings as regression targets.
    """
    with torch.no_grad():
        tgt_text_emb = text_stem(tgt_text_ids)

    l1 = F.l1_loss(pred_text_emb, tgt_text_emb)
    mse = F.mse_loss(pred_text_emb, tgt_text_emb)
    return l1_weight * l1 + mse_weight * mse


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    warmup = 500
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: (
            s / max(1, warmup)
            if s < warmup
            else max(
                0.05,
                0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
            )
        ),
    )


def maybe_load_jepa_checkpoint(model: Decoder, ckpt_path: str | None) -> None:
    if not ckpt_path:
        return
    state = torch.load(ckpt_path, map_location="cpu")
    # Support both raw state_dict and wrapped checkpoints.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  missing keys   : {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def train_one_epoch(
    model: Decoder,
    loader_a,
    loader_c,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch_idx: int,
    epochs: int,
    global_step: int,
) -> tuple[int, float]:
    model.train()
    running = 0.0
    steps = 0
    total_steps = len(loader_a) + len(loader_c)
    iters = [iter(loader_a), iter(loader_c)]

    for i in tqdm(range(total_steps), desc=f"epoch {epoch_idx + 1}/{epochs}"):
        loader_idx = i % 2
        tag = "A2T" if loader_idx == 0 else "T2A"

        try:
            batch = next(iters[loader_idx])
        except StopIteration:
            continue
        if not batch:
            continue

        b = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if loader_idx == 0:
            # ObjA: audio -> text
            z_ctx = model.encode(audio_mel=b["ctx_audio"], pad_mask=b["ctx_pad_mask"])
            z_hat = model.predict(
                z_ctx, b["src_lang"], b["src_mod"], b["tgt_lang"], b["tgt_mod"]
            )
            pred_text = model.text_decoder(
                model.text_stem(b["tgt_text"]),
                z_hat,
                b["tgt_mod"],
                b["tgt_lang"],
            )
            loss = text_reconstruction_loss(pred_text, b["tgt_text"], model.text_stem)
        else:
            # ObjC: text -> audio
            z_ctx = model.encode(text_ids=b["ctx_text"], pad_mask=b["ctx_pad_mask"])
            z_hat = model.predict(
                z_ctx, b["src_lang"], b["src_mod"], b["tgt_lang"], b["tgt_mod"]
            )
            pred_mel = model.audio_decoder(
                z_hat, b["tgt_lang"], target_len=b["tgt_audio"].size(-1)
            )
            loss = mel_reconstruction_loss(pred_mel, b["tgt_audio"])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
        )
        optimizer.step()
        scheduler.step()

        running += loss.item()
        steps += 1
        global_step += 1

        if global_step % LOG_EVERY == 0:
            print(
                f"\nstep {global_step:06d} {tag} "
                f"loss {loss.item():.4f} "
                f"lr {scheduler.get_last_lr()[0]:.2e}"
            )

    avg = running / max(1, steps)
    return global_step, avg


def main():
    parser = argparse.ArgumentParser(description="Train decoder heads on top of MMT-JEPA.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--jepa-ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="decoder_checkpoints")
    parser.add_argument("--freeze-jepa", action="store_true", default=True)
    parser.add_argument("--unfreeze-jepa", dest="freeze_jepa", action="store_false")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device:", device)

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load("tokenizer/tokenizer.model")

    cfg = ModelConfig()
    loader_a = ObjA(sp, cfg).loader(batch_size=args.batch_size, num_workers=args.num_workers)
    loader_c = ObjC(sp, cfg).loader(batch_size=args.batch_size, num_workers=args.num_workers)
    steps_per_epoch = len(loader_a) + len(loader_c)
    total_steps = max(1, steps_per_epoch * args.epochs)
    print(f"Steps/epoch: {steps_per_epoch:,}  total: {total_steps:,}")

    base = MMT_JEPA(cfg)
    model = Decoder(cfg, base, freeze_jepa=args.freeze_jepa).to(device)
    maybe_load_jepa_checkpoint(model, args.jepa_ckpt)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters    : {total:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(optimizer, total_steps=total_steps)

    os.makedirs(args.out_dir, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        step, avg = train_one_epoch(
            model=model,
            loader_a=loader_a,
            loader_c=loader_c,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch_idx=epoch,
            epochs=args.epochs,
            global_step=step,
        )
        elapsed = time.time() - t0
        print(f"epoch {epoch + 1}/{args.epochs}  avg_loss {avg:.4f}  {elapsed:.0f}s")

        ckpt_path = os.path.join(args.out_dir, f"decoder_epoch{epoch + 1:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
