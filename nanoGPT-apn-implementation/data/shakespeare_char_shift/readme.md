# Shakespeare Char Shift Dataset

This is a **shifted domain dataset** for continual learning experiments with APN.

## Purpose

Used as "Domain B" in the continual adaptation protocol:
- Models pretrained on original `shakespeare_char` (Domain A)
- Adapt on this shifted dataset (Domain B)
- Evaluate retention (original domain) vs adaptation (shifted domain)

## Content

Synthetic Shakespeare-style text with:
- **New character names**: Aldric, Bertram, Celestine, Damon, Elara, Felix, Gabrielle, etc.
- **Similar style**: Iambic-ish dialogue, medieval setting
- **Same vocabulary**: Uses identical `stoi`/`itos` from original `shakespeare_char`

## Preparation

```sh
# Requires original shakespeare_char to be prepared first
python data/shakespeare_char/prepare.py

# Then prepare this shifted dataset
python data/shakespeare_char_shift/prepare.py
```

## Files

- `train.bin` - Training data as uint16 token ids
- `val.bin` - Validation data as uint16 token ids  
- `meta.pkl` - Vocabulary (same as shakespeare_char)

## Usage

Used automatically by `continual_apn.py` for the continual learning experiment.
