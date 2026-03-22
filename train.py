import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from model   import MMT_JEPA, ModelConfig
from dataset import ObjA, ObjB, ObjC
from tqdm import tqdm

EPOCHS     = 50
LR         = 3e-4
BATCH_SIZE = 32
LOG_EVERY  = 50
GRAD_CLIP  = 1.0

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print("Using device ", device)
    class _Tok:
        import sentencepiece as spm
        _sp = spm.SentencePieceProcessor()
        _sp.Load("model/tokenizer.model")
        def encode(self, text): return self._sp.Encode(text)

    tok    = _Tok()
    ds     = ConcatDataset(
        ObjB(tok), 
        ObjA(tok), 
        ObjC(tok), 
    )
    print("Total Dataset =", len(ds))

    loader = ds.loader(batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    total  = len(loader) * EPOCHS

    model = MMT_JEPA(ModelConfig()).to(device)
    opt   = torch.optim.AdamW(
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
        t0 = time.time()
        running = 0.0
        for batch in tqdm(loader):
            if not batch: 
                continue
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            z_hat, z_tgt = model(
                ctx_text=b["ctx_text"], ctx_audio=b["ctx_audio"],
                tgt_text=b["tgt_text"], tgt_audio=b["tgt_audio"],
                src_lang=b["src_lang"], src_mod=b["src_mod"],
                tgt_lang=b["tgt_lang"], tgt_mod=b["tgt_mod"],
                ctx_pad_mask=b["ctx_pad_mask"], tgt_pad_mask=b["tgt_pad_mask"],
            )
            loss = F.mse_loss(z_hat, z_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
            sched.step()
            model.ema_step(total_steps=total)
            running += loss.item() 
            step += 1
            if step % LOG_EVERY == 0:
                print(f"\nstep {step:05d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}")
        print(f"epoch {epoch+1}/{EPOCHS}  avg {running/len(loader):.4f}  {time.time()-t0:.0f}s")
        torch.save(model.state_dict(), f"checkpoints/objb_epoch{epoch+1:03d}.pt")