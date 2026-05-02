# MMT-JEPA

A multimodal English ↔ Twi model trained with a LeJEPA-style objective (predictive MSE + SIGReg on encoder pools).

## What it does

Learns a shared latent space across text and audio in both languages by training a predictor to anticipate target representations from context — no reconstruction loss, no cascaded pipeline.

Three training objectives:
- **A** — Audio → Text (both languages)
- **B** — Text → Text (translation)
- **C** — Text → Audio (both languages)

## Files

| File | Purpose |
|---|---|
| `model.py` | `MMT_JEPA` (shared encoder + predictor) |
| `sigreg.py` | SIGReg loss (LeJEPA) |
| `dataset.py` | `ObjA`, `ObjB`, `ObjC` dataset classes |
| `tokenizer.py` | Trains a joint BPE tokenizer on all text data |
| `train.py` | SSL pretraining (all objectives) |
| `train_decoder.py` | Decoder fine-tuning on frozen or tunable JEPA |

## Setup

```bash
pip install torch torchaudio soundfile sentencepiece datasets
```

## Usage

**1. Train the tokenizer**
```bash
python tokenizer.py
# outputs: tokenizer.model, tokenizer.vocab
```

**2. Train the model**
```bash
python train.py
```

Checkpoints saved to `checkpoints/epoch{N}.pt` after each epoch.

## Data

| Objective | Dataset |
|---|---|
| A + C (English audio) | [LibriSpeech train-clean-100](https://huggingface.co/datasets/openslr/librispeech_asr) |
| A + C (Twi audio) | [twi-speech-text-multispeaker-16k](https://huggingface.co/datasets/ghananlpcommunity/twi-speech-text-multispeaker-16k) |
| B (translation) | [twi-english-paragraph-dataset_news](https://huggingface.co/datasets/ghananlpcommunity/twi-english-paragraph-dataset_news) · [english-twi-sentences-non-nouns](https://huggingface.co/datasets/ghananlpcommunity/english-twi-sentences-non-nouns) · [english-twi-nouns-v2](https://huggingface.co/datasets/ghananlpcommunity/english-twi-nouns-v2) |

All datasets load automatically via HuggingFace on first run.

## Model config

Edit `ModelConfig` in `model.py` to change capacity:

```python
d_model        = 512    # embedding dimension
trunk_layers   = 6      # shared transformer depth
vocab_size     = 16_000
n_mels         = 80
sample_rate    = 16_000
sigreg_lambda  = 0.02 # LeJEPA trade-off (TinyMMT_JEPAConfig in config.py for training runs)
```

## Training notes

- L2-normalise pooled predictions and targets before MSE; SIGReg runs on raw pooled ctx/tgt stacks
- Loss: `(1 - λ) · MSE + λ · SIGReg` with `λ = sigreg_lambda`
- Possible `COLLAPSE` log when embedding `std` is tiny or cosine similarity is near 1