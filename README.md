# MMT-JEPA

A multimodal machine translation model for English ↔ Twi using a JEPA (Joint Embedding Predictive Architecture) objective.

## What it does

Learns a shared latent space across text and audio in both languages by training a predictor to anticipate target representations from context — no reconstruction loss, no cascaded pipeline.

Three training objectives:
- **A** — Audio → Text (both languages)
- **B** — Text → Text (translation)
- **C** — Text → Audio (both languages)

## Files

| File | Purpose |
|---|---|
| `model.py` | `MMT_JEPA` model + EMA target encoder |
| `dataset.py` | `ObjA`, `ObjB`, `ObjC` dataset classes |
| `tokenizer.py` | Trains a joint BPE tokenizer on all text data |
| `train.py` | Training loop (all objectives) |
| `train_b.py` | Training loop (Objective B only) |

## Setup

```bash
pip install torch librosa soundfile sentencepiece datasets
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
d_model      = 512    # embedding dimension
trunk_layers = 6      # shared transformer depth
vocab_size   = 16_000
n_mels       = 80
sample_rate  = 16_000
```

## Training notes

- First 5 epochs run text-only (ObjB) to warm up representations before audio is introduced
- L2 normalization applied to both sides before MSE loss to keep scale stable across modalities
- EMA target encoder uses cosine-annealed decay (0.990 → 0.996)
- Collapse logged as `COLLAPSE` when `std < 0.01` or `cos_sim > 0.99`