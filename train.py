import math
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import ModelConfig
from dataset import ObjA, ObjB, ObjC
from logger import TrainingLogger
from model import MMT_JEPA
from sigreg import SIGReg

EPOCHS     = 10
LR         = 3e-4
BATCH_SIZE = 32
LOG_EVERY  = 50
GRAD_CLIP  = 1.0
MAX_SAMPLES = None

if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device:", device)
    # run_log = TrainingLogger.create(name="jepa")
    # print(f"Metrics dir: {run_log.run_path}")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("tokenizer/tokenizer.model")

    cfg = ModelConfig()

    # loader_a = ObjA(sp, cfg, max_samples=MAX_SAMPLES).loader(batch_size=BATCH_SIZE, num_workers=2)
    loader_b = ObjB(sp, cfg, max_samples=MAX_SAMPLES).loader(batch_size=BATCH_SIZE, num_workers=2)
    loader_c = ObjC(sp, cfg, max_samples=MAX_SAMPLES).loader(batch_size=BATCH_SIZE, num_workers=2)

    loaders         = [loader_b, loader_c]
    steps_per_epoch = sum(len(loader) for loader in loaders)
    total           = steps_per_epoch * EPOCHS
    print(f"Steps/epoch: {steps_per_epoch:,}  total: {total:,}")

    model = MMT_JEPA(cfg).to(device)
    sigreg = SIGReg(knots=cfg.sigreg_knots).to(device)
    lam = cfg.sigreg_lambda
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01, betas=(0.9, 0.95),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: (
        s / max(1, 500) if s < 500
        else max(0.05, 0.5 * (1 + math.cos(math.pi * (s - 500) / max(1, total - 500))))
    ))

    step = 0
    model.train()

    for epoch in range(EPOCHS):
        t0       = time.time()
        running  = 0.0
        iters    = [iter(loader) for loader in loaders]

        for idx in tqdm(range(steps_per_epoch), desc=f"epoch {epoch+1}/{EPOCHS}"):
            loader_idx = idx % len(loaders)
            try:
                batch = next(iters[loader_idx])
            except StopIteration:
                iters[loader_idx] = iter(loaders[loader_idx])
                try:
                    batch = next(iters[loader_idx])
                except StopIteration:
                    continue
            if not batch:
                continue

            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

            opt.zero_grad(set_to_none=True)
            z_hat, z_tgt, sigreg_proj = model(
                ctx_text=b["ctx_text"], ctx_audio=b["ctx_audio"],
                tgt_text=b["tgt_text"], tgt_audio=b["tgt_audio"],
                src_lang=b["src_lang"], src_mod=b["src_mod"],
                tgt_lang=b["tgt_lang"], tgt_mod=b["tgt_mod"],
                ctx_pad_mask=b["ctx_pad_mask"], tgt_pad_mask=b["tgt_pad_mask"],
            )
            pred_loss = F.mse_loss(z_hat, z_tgt)
            sigreg_loss = sigreg(sigreg_proj)
            loss = (1.0 - lam) * pred_loss + lam * sigreg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
            sched.step()

            running += loss.item()
            step    += 1

            if step % LOG_EVERY == 0:
                with torch.no_grad():
                    std = z_hat.std(dim=0).mean().item()
                    cos = F.cosine_similarity(z_hat, z_tgt, dim=-1).mean().item()
                flag = " COLLAPSE" if std < 0.01 or cos > 0.99 else ""
                print(
                    f"\nstep {step:05d} {'BC'[loader_idx]}  loss {loss.item():.4f}  "
                    f"pred {pred_loss.item():.4f}  sigreg {sigreg_loss.item():.4f}  "
                    f"std {std:.3f}  cos {cos:.3f}  lr {sched.get_last_lr()[0]:.2e}{flag}"
                )
                # run_log.log_step(
                #     {
                #         "step": step,
                #         "epoch": epoch + 1,
                #         "branch": "ABC"[loader_idx],
                #         "loss": loss.item(),
                #         "pred_loss": pred_loss.item(),
                #         "sigreg_loss": sigreg_loss.item(),
                #         "std": std,
                #         "cos": cos,
                #         "lr": sched.get_last_lr()[0],
                #     }
                # )

        avg = running / max(1, steps_per_epoch)
        print(f"epoch {epoch+1}/{EPOCHS}  avg_loss {avg:.4f}  {time.time()-t0:.0f}s")
        # run_log.log_epoch(epoch + 1, avg, time.time() - t0)
        torch.save(model.state_dict(), f"checkpoints/epoch{epoch+1:03d}.pt")

    # run_log.close()
