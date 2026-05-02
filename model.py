from torch import Tensor, nn
import torch
import torch.nn.functional as F
import math
from config import ModelConfig
from sigreg import SIGReg



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
    """Mel front-end: two convs + avg-pool (~×2 downsample). Avoids strided-conv MPS backward bugs."""

    def __init__(self, n_mels: int, d_model: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, 3, padding=1)
        self.act   = nn.SiLU()
        self.conv2 = nn.Conv1d(d_model, d_model, 3, stride=1, padding=1)
        self.down  = nn.AvgPool1d(kernel_size=2, stride=2)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_mels, T)
        x = self.act(self.conv1(x.contiguous())).contiguous()
        x = self.conv2(x).contiguous()
        x = self.down(x)
        return self.norm(x.transpose(1, 2).contiguous())


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
        return self.encoder(x, src_key_padding_mask=key_padding_mask).contiguous()


class Predictor(nn.Module):
    """Context sequence + lang/mod tokens → predicted target sequence in latent space."""

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
        return self.proj_out(x[:, 1:, :]).contiguous()         # (B, T, d_model)


class MMT_JEPA(nn.Module):
    """LeJEPA-style multimodal JEPA: shared encoder, predictor, MSE on pools + SIGReg on ctx/tgt pools."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.audio_stem = AudioStem(cfg.n_mels, cfg.d_model)
        self.text_stem  = TextStem(cfg.d_model, cfg.vocab_size)
        self.pe         = SinusoidalPE(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.trunk      = Shunt(cfg)
        self.predictor  = Predictor(cfg)

    def encode(
        self,
        text_ids:  Tensor | None = None,  # (B, L)
        audio_mel: Tensor | None = None,  # (B, n_mels, T)
        pad_mask:  Tensor | None = None,  # (B, S) bool — True where padded
    ) -> Tensor:
        """Encoder: stem → sinusoidal PE → shared trunk (text XOR audio)."""
        assert (text_ids is None) != (audio_mel is None), \
            "Provide exactly one of text_ids or audio_mel."
        if audio_mel is not None:
            audio_mel = audio_mel.contiguous()
        x = self.text_stem(text_ids) if text_ids is not None else self.audio_stem(audio_mel)
        x = self.pe(x)
        return self.trunk(x, key_padding_mask=pad_mask)

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
            out = (z * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
            return out.contiguous()
        return z.mean(dim=1).contiguous()

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
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns normalised (z_hat, z_tgt) and raw ``sigreg_proj`` (2, B, d) for SIGReg."""
        z_ctx = self.encode(ctx_text, ctx_audio, ctx_pad_mask)
        z_ctx_raw = self._pool(z_ctx, ctx_pad_mask)

        z_seq = self.predictor(z_ctx, src_lang, src_mod, tgt_lang, tgt_mod)
        z_hat = self._pool(z_seq, ctx_pad_mask)

        z_full = self.encode(tgt_text, tgt_audio, tgt_pad_mask)
        z_tgt_raw = self._pool(z_full, tgt_pad_mask)

        sigreg_proj = torch.stack([z_ctx_raw, z_tgt_raw], dim=0)

        z_hat = F.normalize(z_hat, dim=-1).contiguous()
        z_tgt = F.normalize(z_tgt_raw, dim=-1).contiguous()

        return z_hat, z_tgt, sigreg_proj


if __name__ == "__main__":
    cfg   = ModelConfig()
    model = MMT_JEPA(cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nTrainable parameters : {trainable:,}")

    B, L, T = 32, 16, 128
    dummy_text  = torch.randint(0, cfg.vocab_size, (B, L))
    dummy_audio = torch.randn(B, cfg.n_mels, T)
    src_lang = torch.zeros(B, dtype=torch.long)   # eng = 0
    tgt_lang = torch.ones(B,  dtype=torch.long)   # twi = 1
    src_mod  = torch.zeros(B, dtype=torch.long)   # text = 0
    tgt_mod  = torch.ones(B,  dtype=torch.long)   # audio = 1

    z_hat, z_tgt, sigreg_proj = model(
        ctx_text=dummy_text, ctx_audio=None,
        tgt_text=None,       tgt_audio=dummy_audio,
        src_lang=src_lang,   src_mod=src_mod,
        tgt_lang=tgt_lang,   tgt_mod=tgt_mod,
    )
    lam = cfg.sigreg_lambda
    pred = F.mse_loss(z_hat, z_tgt)
    sig = SIGReg(knots=cfg.sigreg_knots)
    reg = sig(sigreg_proj)
    loss = (1.0 - lam) * pred + lam * reg
    loss.backward()

    print(f"\nSmoke-test pred/reg/total : {pred.item():.4f} / {reg.item():.4f} / {loss.item():.4f}")
    print(f"z_hat shape          : {z_hat.shape}")
    print(f"sigreg_proj shape    : {sigreg_proj.shape}")