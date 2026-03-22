from torch import Tensor, nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import math
import copy


def jepa_loss(z_hat: Tensor, z_tgt: Tensor, temp: float = 0.1, lam: float = 0.5) -> Tensor:
    """
    Combined MSE + InfoNCE loss as used in VL-JEPA.
    Both inputs must already be L2-normalised.

    lam : weight on InfoNCE (0 = pure MSE, 1 = pure InfoNCE)
    temp: InfoNCE temperature — lower = sharper, more collapse-resistant
    """
    # alignment (MSE on unit vectors = 2 - 2*cos, bounded [0,4])
    mse = F.mse_loss(z_hat, z_tgt)

    # uniformity via InfoNCE
    # logits: (B, B) — diagonal entries are positives
    logits = torch.matmul(z_hat, z_tgt.T) / temp   # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    nce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    return (1 - lam) * mse + lam * nce


@dataclass
class ModelConfig:
    d_model:      int   = 512
    n_heads:      int   = 8
    trunk_layers: int   = 6
    pred_layers:  int   = 12      # narrow predictor
    pred_dim:     int   = 512    # d_model // 2
    vocab_size:   int   = 16_000
    n_mels:       int   = 80
    n_langs:      int   = 2      # eng, twi
    n_mods:       int   = 2      # text, audio
    dropout:      float = 0.15
    ema_decay:    float = 0.996
    max_seq_len:  int   = 900   # upper bound for PE cache
    sample_rate:  int   = 16_000 # audio sample rate


class SinusoidalPE(nn.Module):
    """
    Classic fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Registered as a buffer so it moves with .to(device) but is never
    treated as a learnable parameter.  The cache is built once up to
    max_seq_len and sliced at forward time, so variable-length sequences
    (text tokens AND downsampled audio frames) are both handled without
    recomputation.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # build (max_seq_len, d_model) table once
        pe    = torch.zeros(max_seq_len, d_model)
        pos   = torch.arange(max_seq_len).unsqueeze(1)          # (L, 1)
        denom = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )                                                        # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)

        self.register_buffer("pe", pe.unsqueeze(0))             # (1, L, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AudioStem(nn.Module):
    def __init__(self, n_mels: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, d_model, 3, padding=1),            # (B, d, T)
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 3, stride=2, padding=1), # (B, d, T//2)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_mels, T)
        out = self.net(x)                        # (B, d_model, T//2)
        return self.norm(out.transpose(1, 2))    # (B, T//2, d_model)


class TextStem(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L) token ids
        return self.norm(self.embed(x))          # (B, L, d_model)


class Shunt(nn.Module):
    """Shared transformer trunk. Both modalities enter here as (B, T, d_model)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,          # pre-norm: more stable for low-resource
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.trunk_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        # x:    (B, T, d_model)
        # mask: (B, T) bool — True where padded
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class Predictor(nn.Module):
    """
    Narrow JEPA predictor.  Receives the context encoding z_ctx and four
    conditioning tokens (src_lang, src_mod, tgt_lang, tgt_mod), and
    predicts the target encoding z_tgt in the shared latent space.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        pd = cfg.pred_dim

        self.lang_emb = nn.Embedding(cfg.n_langs, pd)
        self.mod_emb  = nn.Embedding(cfg.n_mods,  pd)

        self.proj_in  = nn.Linear(cfg.d_model, pd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pd,
            nhead=max(1, pd // 64),
            dim_feedforward=pd * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder  = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.pred_layers,
            enable_nested_tensor=False,
        )
        self.proj_out = nn.Linear(pd, cfg.d_model)

    def forward(
        self,
        z_ctx:    Tensor,   # (B, T, d_model)
        src_lang: Tensor,   # (B,) int
        src_mod:  Tensor,   # (B,) int
        tgt_lang: Tensor,   # (B,) int
        tgt_mod:  Tensor,   # (B,) int
    ) -> Tensor:
        cond = (
            self.lang_emb(src_lang)
            + self.mod_emb(src_mod)
            + self.lang_emb(tgt_lang)
            + self.mod_emb(tgt_mod)
        ).unsqueeze(1)                                          # (B, 1, pd)

        x = torch.cat([cond, self.proj_in(z_ctx)], dim=1)     # (B, T+1, pd)
        x = self.encoder(x)
        return self.proj_out(x[:, 1:, :])                      # (B, T, d_model)


class MMT_JEPA(nn.Module):
    """
    Full model: stems → PE → shared trunk → predictor.

    The EMA target encoder is an internal shadow copy of the online encoder
    (audio_stem + text_stem + pe + trunk).  Its weights are never touched by
    the optimiser — only by ema_step().  It produces the stop-gradient target
    z_tgt used in the JEPA L2 loss.

    EMA schedule — decay is annealed with a cosine ramp from ema_decay_base
    up to ema_decay over total_steps.  This gives the target encoder more
    freedom to move early in training (when the online encoder is still far
    from useful representations) and locks it in tightly later.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.audio_stem = AudioStem(cfg.n_mels, cfg.d_model)
        self.text_stem  = TextStem(cfg.d_model, cfg.vocab_size)
        self.pe         = SinusoidalPE(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.trunk      = Shunt(cfg)
        self.predictor  = Predictor(cfg)

        self.target_audio_stem = copy.deepcopy(self.audio_stem)
        self.target_text_stem  = copy.deepcopy(self.text_stem)
        self.target_pe         = copy.deepcopy(self.pe)
        self.target_trunk      = copy.deepcopy(self.trunk)
        self._set_target_grad(False)

        self.register_buffer("ema_step_count", torch.tensor(0, dtype=torch.long))

    def _set_target_grad(self, requires: bool) -> None:
        for p in self._target_params():
            p.requires_grad_(requires)

    def _target_params(self):
        """Yield all parameters that belong to the target encoder."""
        for module in (
            self.target_audio_stem,
            self.target_text_stem,
            self.target_pe,
            self.target_trunk,
        ):
            yield from module.parameters()

    def _online_params(self):
        """Yield online encoder parameters in the same order as _target_params."""
        for module in (self.audio_stem, self.text_stem, self.pe, self.trunk):
            yield from module.parameters()

    def _cosine_decay(self, total_steps: int) -> float:
        """
        Cosine-annealed EMA decay.
        Starts at ema_decay_base (≈0.99) and rises to cfg.ema_decay (0.996).
        Keeps the target responsive early, then stabilises it.
        """
        base  = max(0.0, self.cfg.ema_decay - 0.006)   # e.g. 0.990
        peak  = self.cfg.ema_decay                      # 0.996
        step  = min(int(self.ema_step_count), total_steps)
        cos_v = math.cos(math.pi * step / total_steps)  # 1 → -1
        return peak - (peak - base) * (cos_v + 1) / 2  # base → peak

    @torch.no_grad()
    def ema_step(self, total_steps: int = 100_000) -> float:
        """
        Update target encoder weights with exponential moving average.

        Call once per training step, after loss.backward() and optimiser.step().
        Returns the decay value used (useful for logging).

        Args:
            total_steps: total expected training steps across the current phase,
                         used for cosine decay scheduling.
        """
        decay = self._cosine_decay(total_steps)
        for p_tgt, p_online in zip(self._target_params(), self._online_params()):
            p_tgt.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)
        self.ema_step_count.add_(1)
        return decay

    def encode(
        self,
        text_ids:  Tensor | None = None,  # (B, L)
        audio_mel: Tensor | None = None,  # (B, n_mels, T)
        pad_mask:  Tensor | None = None,  # (B, S) bool — True where padded
    ) -> Tensor:
        """Online encoder: stem → sinusoidal PE → shared trunk."""
        assert (text_ids is None) != (audio_mel is None), \
            "Provide exactly one of text_ids or audio_mel."
        x = self.text_stem(text_ids) if text_ids is not None else self.audio_stem(audio_mel)
        x = self.pe(x)
        return self.trunk(x, key_padding_mask=pad_mask)

    @torch.no_grad()
    def encode_target(
        self,
        text_ids:  Tensor | None = None,
        audio_mel: Tensor | None = None,
        pad_mask:  Tensor | None = None,
    ) -> Tensor:
        """
        Target encoder: identical path through the EMA shadow copies.

        Always runs under no_grad — the stop-gradient on z_tgt is structural,
        not a .detach() call that can be accidentally removed.
        """
        assert (text_ids is None) != (audio_mel is None), \
            "Provide exactly one of text_ids or audio_mel."
        x = (
            self.target_text_stem(text_ids)
            if text_ids is not None
            else self.target_audio_stem(audio_mel)
        )
        x = self.target_pe(x)
        return self.target_trunk(x, key_padding_mask=pad_mask)

    @staticmethod
    def _pool(z: Tensor, pad_mask: Tensor | None) -> Tensor:
        """
        Mean-pool a sequence (B, T, d) → (B, d), respecting padding.

        When pad_mask is provided (True = padded position), masked positions
        are zeroed before summing so they don't contribute to the mean.
        """
        if pad_mask is not None:
            # pad_mask: (B, T) bool, True where padded
            keep = (~pad_mask).unsqueeze(-1).float()   # (B, T, 1)
            return (z * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        return z.mean(dim=1)

    def forward(
        self,
        ctx_text:     Tensor | None,
        ctx_audio:    Tensor | None,
        tgt_text:     Tensor | None,
        tgt_audio:    Tensor | None,
        src_lang:     Tensor,
        src_mod:      Tensor,
        tgt_lang:     Tensor,
        tgt_mod:      Tensor,
        ctx_pad_mask: Tensor | None = None,
        tgt_pad_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Single forward pass for one JEPA objective.

        Cross-modal objectives (text→audio, audio→text) produce sequences of
        different lengths, so both sides are mean-pooled to a single vector
        before the loss.  Same-modality objectives pool identically — no
        special casing needed in the training loop.

        Returns:
            z_hat : (B, d_model)  predicted target representation  [requires grad]
            z_tgt : (B, d_model)  EMA target representation        [no grad]

        Loss in the training loop:
            loss = F.mse_loss(z_hat, z_tgt)
        """
        z_ctx = self.encode(ctx_text, ctx_audio, ctx_pad_mask)
        z_seq = self.predictor(z_ctx, src_lang, src_mod, tgt_lang, tgt_mod)
        z_hat = self._pool(z_seq, ctx_pad_mask)

        z_full = self.encode_target(tgt_text, tgt_audio, tgt_pad_mask)
        z_tgt  = self._pool(z_full, tgt_pad_mask)

        # L2-normalise before loss — keeps MSE scale-invariant across modalities
        z_hat = F.normalize(z_hat, dim=-1)
        z_tgt = F.normalize(z_tgt, dim=-1)

        return z_hat, z_tgt


if __name__ == "__main__":
    cfg   = ModelConfig()
    model = MMT_JEPA(cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(model)
    print(f"\nTrainable parameters : {trainable:,}")
    print(f"Frozen (EMA target)  : {frozen:,}")

    # --- smoke test: one JEPA forward pass + EMA update ---
    B, L, T = 32, 16, 128
    dummy_text  = torch.randint(0, cfg.vocab_size, (B, L))
    dummy_audio = torch.randn(B, cfg.n_mels, T)
    src_lang = torch.zeros(B, dtype=torch.long)   # eng = 0
    tgt_lang = torch.ones(B,  dtype=torch.long)   # twi = 1
    src_mod  = torch.zeros(B, dtype=torch.long)   # text = 0
    tgt_mod  = torch.ones(B,  dtype=torch.long)   # audio = 1

    z_hat, z_tgt = model(
        ctx_text=dummy_text, ctx_audio=None,
        tgt_text=None,       tgt_audio=dummy_audio,
        src_lang=src_lang,   src_mod=src_mod,
        tgt_lang=tgt_lang,   tgt_mod=tgt_mod,
    )
    loss  = F.mse_loss(z_hat, z_tgt)
    loss.backward()

    decay = model.ema_step(total_steps=50_000)
    print(f"\nSmoke-test loss      : {loss.item():.4f}")
    print(f"EMA decay used       : {decay:.6f}")
    print(f"EMA step count       : {int(model.ema_step_count)}")
    print(f"z_hat shape          : {z_hat.shape}  (B, d_model — mean-pooled)")
    print(f"z_tgt shape          : {z_tgt.shape}  (B, d_model — mean-pooled)")