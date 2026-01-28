#!/usr/bin/env python3
"""
Generate samples from trained baseline and APN checkpoints.

This script loads checkpoints and generates text samples, saving them to files.
Uses the same generation logic as sample.py but with file output.

Usage:
    python generate_samples.py [--device DEVICE] [--num_samples N] [--max_new_tokens N]

Outputs:
    out-shakespeare-char/samples/sample_*.txt
    out-shakespeare-char-apn/samples/sample_*.txt
"""

import os
import sys
import pickle
import argparse
from contextlib import nullcontext
from datetime import datetime

import torch
from model import GPTConfig, GPT


def load_model_and_meta(out_dir, device):
    """Load model checkpoint and metadata."""
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Create model
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(device)
    
    # Load meta for encode/decode
    encode = None
    decode = None
    
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        if os.path.exists(meta_path):
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
    
    if encode is None:
        raise RuntimeError("Could not load character encoding from meta.pkl")
    
    return model, encode, decode, checkpoint


def generate_samples(model, encode, decode, prompt, num_samples, max_new_tokens,
                     temperature, top_k, device, seed=1337):
    """Generate text samples from the model."""
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed(seed)
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # Determine dtype
    if device_type == 'cuda' and torch.cuda.is_bf16_supported():
        ptdtype = torch.bfloat16
    elif device_type == 'cuda':
        ptdtype = torch.float16
    else:
        ptdtype = torch.float32
    
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Encode prompt
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    samples = []
    
    with torch.no_grad():
        with ctx:
            for i in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                text = decode(y[0].tolist())
                samples.append(text)
                print(f"  Generated sample {i+1}/{num_samples} ({len(text)} chars)")
    
    return samples


def save_samples(samples, out_dir, prefix="sample"):
    """Save samples to individual text files."""
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    paths = []
    for i, sample in enumerate(samples):
        filename = f"{prefix}_{i+1:02d}.txt"
        filepath = os.path.join(samples_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(sample)
        paths.append(filepath)
    
    # Also save all samples to a single file
    all_filepath = os.path.join(samples_dir, f"{prefix}_all.txt")
    with open(all_filepath, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(samples):
            f.write(f"=== Sample {i+1} ===\n\n")
            f.write(sample)
            f.write("\n\n" + "="*40 + "\n\n")
    paths.append(all_filepath)
    
    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained models")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu, cuda, mps). Auto-detected if not specified.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate per model")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                        help="Max tokens to generate per sample (default: 500)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=200,
                        help="Top-k sampling (default: 200)")
    parser.add_argument("--prompt", type=str, default="\n",
                        help="Starting prompt for generation")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for reproducibility")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Only generate from baseline model")
    parser.add_argument("--apn_only", action="store_true",
                        help="Only generate from APN model")
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Generation settings: max_tokens={args.max_new_tokens}, temp={args.temperature}, top_k={args.top_k}")
    print(f"Prompt: {repr(args.prompt)}")
    print()
    
    # Define models to sample from
    models_to_sample = []
    
    if not args.apn_only:
        models_to_sample.append({
            "name": "Baseline (MLP)",
            "out_dir": "out-shakespeare-char",
            "prefix": "baseline"
        })
    
    if not args.baseline_only:
        models_to_sample.append({
            "name": "APN",
            "out_dir": "out-shakespeare-char-apn",
            "prefix": "apn"
        })
    
    results = []
    
    for model_info in models_to_sample:
        print("=" * 60)
        print(f"Generating samples from {model_info['name']}")
        print("=" * 60)
        
        out_dir = model_info['out_dir']
        
        # Check if checkpoint exists
        if not os.path.exists(os.path.join(out_dir, "ckpt.pt")):
            print(f"  WARNING: Checkpoint not found in {out_dir}, skipping...")
            print()
            continue
        
        try:
            # Load model
            model, encode, decode, checkpoint = load_model_and_meta(out_dir, device)
            
            # Report model info
            model_args = checkpoint.get('model_args', {})
            use_apn = model_args.get('use_apn', False)
            print(f"  Model type: {'APN' if use_apn else 'MLP'}")
            if use_apn:
                print(f"  APN config: K={model_args.get('apn_K')}, k={model_args.get('apn_k')}, r={model_args.get('apn_r')}")
            
            val_loss = checkpoint.get('best_val_loss')
            if val_loss:
                print(f"  Best val loss: {val_loss:.4f}")
            print()
            
            # Generate samples
            print(f"Generating {args.num_samples} samples...")
            samples = generate_samples(
                model, encode, decode,
                prompt=args.prompt,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
                seed=args.seed
            )
            
            # Save samples
            saved_paths = save_samples(samples, out_dir, prefix=model_info['prefix'])
            print(f"\nSaved {len(samples)} samples to {out_dir}/samples/")
            
            results.append({
                "name": model_info['name'],
                "out_dir": out_dir,
                "num_samples": len(samples),
                "files": saved_paths
            })
            
            # Print first sample preview
            print("\n--- Sample Preview (first 300 chars) ---")
            preview = samples[0][:300] + ("..." if len(samples[0]) > 300 else "")
            print(preview)
            print("-" * 40)
            
            # Clean up
            del model
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Output directory: {r['out_dir']}/samples/")
        print(f"  Samples generated: {r['num_samples']}")
    
    if not results:
        print("\nNo samples were generated. Make sure to train models first:")
        print("  python train.py config/train_shakespeare_char.py")
        print("  python train.py config/train_shakespeare_char_apn.py")
    
    return results


if __name__ == "__main__":
    main()
