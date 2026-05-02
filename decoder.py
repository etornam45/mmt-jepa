import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model import MMT_JEPA, ModelConfig
from typing import Literal


LANG_TO_ID = {"eng": 0, "twi": 1}
MOD_TO_ID = {"text": 0, "audio": 1}


def build_io_ids(
    batch_size: int,
    device: torch.device,
    src_lang: Literal["eng", "twi"],
    tgt_lang: Literal["eng", "twi"],
    src_mod: Literal["text", "audio"],
    tgt_mod: Literal["text", "audio"],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Build batched id tensors for source/target language and modality labels.
    """
    try:
        src_lang_id = LANG_TO_ID[src_lang]
        tgt_lang_id = LANG_TO_ID[tgt_lang]
        src_mod_id = MOD_TO_ID[src_mod]
        tgt_mod_id = MOD_TO_ID[tgt_mod]
    except KeyError as exc:
        raise ValueError(f"Unknown label: {exc.args[0]}") from exc

    return (
        torch.full((batch_size,), src_lang_id, dtype=torch.long, device=device),
        torch.full((batch_size,), tgt_lang_id, dtype=torch.long, device=device),
        torch.full((batch_size,), src_mod_id, dtype=torch.long, device=device),
        torch.full((batch_size,), tgt_mod_id, dtype=torch.long, device=device),
    )


def mel_reconstruction_loss(
    pred_mel: Tensor, gt_mel: Tensor, l1_weight: float = 1.0, mse_weight: float = 0.5
) -> Tensor:
    """
    Reconstruction loss for decoder fine-tuning.
    Expects pred and gt in (B, n_mels, T). Time is aligned if needed.
    """
    if pred_mel.size(-1) != gt_mel.size(-1):
        pred_mel = F.interpolate(
            pred_mel, size=gt_mel.size(-1), mode="linear", align_corners=False
        )

    l1 = F.l1_loss(pred_mel, gt_mel)
    mse = F.mse_loss(pred_mel, gt_mel)
    return l1_weight * l1 + mse_weight * mse


class TextDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.dec_dim,
            batch_first=True,
            dim_feedforward=cfg.dec_dim * 4,
            dropout=cfg.dropout,
            nhead=cfg.n_heads,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.dec_layers)

        self.to_lang = nn.Embedding(cfg.n_langs, cfg.dec_dim)
        self.to_mod  = nn.Embedding(cfg.n_mods,  cfg.dec_dim)
        self.lm_head = nn.Linear(cfg.dec_dim, cfg.vocab_size, bias=False)

    def forward(
        self,
        tgt_text_emb: Tensor,   # (B, L, d_model) — shifted-right target embeddings
        z_hat: Tensor,          # (B, S, d_model) — encoder memory
        out_mod: Tensor,        # (B,) int
        out_lang: Tensor,       # (B,) int
    ) -> Tensor:                # (B, L, vocab_size) logits
        L = tgt_text_emb.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=tgt_text_emb.device
        )
        lang = self.to_lang(out_lang).unsqueeze(1)
        mod  = self.to_mod(out_mod).unsqueeze(1)
        tgt  = lang + mod + tgt_text_emb               # (B, L, d_model)
        out  = self.decoder(tgt, z_hat, tgt_mask=causal_mask)  # (B, L, d_model)
        return self.lm_head(out)                        # (B, L, vocab_size)


class AudioDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.to_lang = nn.Embedding(cfg.n_langs, cfg.dec_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.dec_dim,
            batch_first=True,
            dim_feedforward=cfg.dec_dim * 4,
            dropout=cfg.dropout,
            nhead=cfg.n_heads,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=cfg.dec_layers)
        self.upsample = nn.ConvTranspose1d(
            in_channels=cfg.dec_dim,
            out_channels=cfg.dec_dim,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.to_mel = nn.Conv1d(cfg.dec_dim, cfg.n_mels, kernel_size=1)

    def forward(
        self, z_hat: Tensor, out_lang: Tensor | None, target_len: int | None = None
    ) -> Tensor:
        if out_lang is not None:
            z_hat = self.to_lang(out_lang).unsqueeze(1) + z_hat  # (B, T, d_model)

        x = self.decoder(z_hat)  # (B, T, d_model)
        x = self.upsample(x.transpose(1, 2))  # (B, d_model, ~2T)
        mel = self.to_mel(x)  # (B, n_mels, ~2T)

        if target_len is not None and mel.size(-1) != target_len:
            mel = F.interpolate(
                mel, size=target_len, mode="linear", align_corners=False
            )

        return mel


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig, mmt_jepa: MMT_JEPA, freeze_jepa=True):
        super().__init__()
        self.cfg = cfg

        self.audio_stem = mmt_jepa.audio_stem
        self.text_stem = mmt_jepa.text_stem
        self.pe = mmt_jepa.pe
        self.trunk = mmt_jepa.trunk
        self.predictor = mmt_jepa.predictor

        self.text_decoder = TextDecoder(cfg)
        self.audio_decoder = AudioDecoder(cfg)

        if freeze_jepa:
            self.audio_stem.requires_grad_(False)
            self.text_stem.requires_grad_(False)
            self.pe.requires_grad_(False)
            self.trunk.requires_grad_(False)
            self.predictor.requires_grad_(False)


    def encode(
        self,
        text_ids: Tensor | None = None,   # (B, L)
        audio_mel: Tensor | None = None,  # (B, n_mels, T)
        pad_mask: Tensor | None = None,   # (B, S) bool — True where padded
    ) -> Tensor:
        """Online encoder: stem → sinusoidal PE → shared trunk."""
        assert (text_ids is None) != (audio_mel is None), (
            "Provide exactly one of text_ids or audio_mel."
        )
        x = (
            self.text_stem(text_ids)
            if text_ids is not None
            else self.audio_stem(audio_mel)
        )
        x = self.pe(x)
        return self.trunk(x, key_padding_mask=pad_mask)

    def predict(
        self,
        z_ctx: Tensor,      # (B, T, d_model)
        src_lang: Tensor,   # (B,) int
        src_mod: Tensor,    # (B,) int
        tgt_lang: Tensor,   # (B,) int
        tgt_mod: Tensor,    # (B,) int
    ) -> Tensor:
        z_seq = self.predictor(z_ctx, src_lang, src_mod, tgt_lang, tgt_mod)
        return z_seq

    def forward(
        self,
        src_lang: Tensor,               # (B,) int
        src_mod: Tensor,                # (B,) int
        tgt_lang: Tensor,               # (B,) int
        tgt_mod: Tensor,                # (B,) int
        text_ids: Tensor | None = None,      # (B, L) — source text ids (encoder input)
        audio_mel: Tensor | None = None,     # (B, n_mels, T) — source audio
        pad_mask: Tensor | None = None,      # (B, S) bool — True where padded
        tgt_text_ids: Tensor | None = None,  # (B, L) — target text ids (decoder input)
    ) -> Tensor:
        z_ctx = self.encode(text_ids, audio_mel, pad_mask)
        z_hat = self.predict(z_ctx, src_lang, src_mod, tgt_lang, tgt_mod)

        tgt_mod_id = (
            int(tgt_mod[0].item()) if isinstance(tgt_mod, Tensor) else int(tgt_mod)
        )

        if tgt_mod_id == 0:
            if tgt_text_ids is None:
                raise ValueError("tgt_text_ids is required when decoding target text.")
            # Shift right for teacher forcing during training: feed ids[:-1], predict ids[1:]
            # In inference (eval mode), feed the full sequence to get the next token prediction
            if self.training:
                tgt_in = self.text_stem(tgt_text_ids[:, :-1])          # (B, L-1, d_model)
            else:
                tgt_in = self.text_stem(tgt_text_ids)                  # (B, L, d_model)

            return self.text_decoder(tgt_in, z_hat, tgt_mod, tgt_lang)  # (B, L, vocab_size)

        if tgt_mod_id == 1:
            target_len = audio_mel.size(-1) if audio_mel is not None else None
            return self.audio_decoder(z_hat, tgt_lang, target_len=target_len)

        raise ValueError(f"Unsupported target modality id: {tgt_mod_id}")


if __name__ == "__main__":
    cfg = ModelConfig()
    device = torch.device("mps")
    mmt_jepa = MMT_JEPA(cfg)
    mmt_jepa.load_state_dict(torch.load("checkpoint/mmt_jepa_10.pt", map_location=device))

    decoder = Decoder(cfg, mmt_jepa, freeze_jepa=True).to(device)
    trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in decoder.parameters())
    print(f"\nTrainable parameters : {trainable:,}")
    print(f"Total parameters     : {total:,}")

    B, L, T = 32, 160, 128
    dummy_audio   = torch.randn(B, cfg.n_mels, T).to(device)
    dummy_tgt_txt = torch.randint(0, cfg.vocab_size, (B, L)).to(device)
    
    src_lang, tgt_lang, src_mod, tgt_mod = build_io_ids(
        batch_size=B,
        device=device,
        src_lang="eng",
        tgt_lang="twi",
        src_mod="audio",
        tgt_mod="audio",
    )

    text_pred = decoder.forward(
        src_lang,
        src_mod,
        tgt_lang=tgt_lang,
        tgt_mod=tgt_mod,
        audio_mel=dummy_audio,
        tgt_text_ids=dummy_tgt_txt,
    )
    print(f"text_pred shape : {text_pred.shape}")  # (B, L, dec_dim)