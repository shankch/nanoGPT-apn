#!/usr/bin/env python3
"""
Continual Adaptation Experiment: APN vs Baseline

Protocol (matching the APN paper):
1. Pretrain: Use existing checkpoints from shakespeare_char training
2. Domain B: Load shifted dataset (shakespeare_char_shift) 
3. Adaptation:
   - Baseline: fine-tune ALL parameters for N iterations
   - APN: freeze all except patch parameters (U, a, b, V; optionally prototypes P)
4. Evaluate:
   - Retention: val loss on original shakespeare_char
   - Adaptation: val loss on shifted dataset

Usage:
    python continual_apn.py [--adapt_iters N] [--device DEVICE]

Outputs:
    out-continual/results.json
    out-continual/results.md
"""

import os
import sys
import json
import time
import math
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT, APN


# ============================================================================
# Data loading (adapted from train.py)
# ============================================================================

def get_batch(data_dir, split, batch_size, block_size, device, device_type):
    """Get a batch of data from train.bin or val.bin."""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y


@torch.no_grad()
def estimate_loss(model, data_dir, batch_size, block_size, device, device_type, ctx, eval_iters=50):
    """Estimate loss on train and val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_dir, split, batch_size, block_size, device, device_type)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ============================================================================
# Model loading and parameter freezing
# ============================================================================

def load_checkpoint(out_dir, device):
    """Load a pretrained checkpoint."""
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    return model, model_args, checkpoint


def freeze_all_except_apn_patches(model, freeze_prototypes=True):
    """
    Freeze all parameters except APN patch parameters.
    
    Trainable parameters after freezing:
    - U (per-patch decoders)
    - a, b (per-patch gates)
    - V (shared code projection)
    - Optionally: prototypes P (if freeze_prototypes=False)
    
    Frozen parameters:
    - Attention weights
    - Embeddings
    - LayerNorms
    - Output head
    - Optionally: prototypes P (if freeze_prototypes=True)
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze APN-specific parameters
    trainable_count = 0
    frozen_count = 0
    
    for name, param in model.named_parameters():
        # Check if this is an APN parameter we want to train
        is_apn_trainable = False
        
        if '.mlp.' in name:
            # APN parameters to unfreeze: U, a, b, V
            if any(f'.{p}' in name or name.endswith(f'.{p}') for p in ['U', 'a', 'b', 'V']):
                is_apn_trainable = True
            # Optionally unfreeze prototypes
            elif '.prototypes' in name and not freeze_prototypes:
                is_apn_trainable = True
        
        if is_apn_trainable:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()
    
    return trainable_count, frozen_count


def get_optimizer(model, learning_rate, weight_decay, device_type):
    """Configure optimizer for continual adaptation."""
    # Get parameters that require grad
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # Weight decay for 2D+ params, no decay for 1D
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.99))
    return optimizer


# ============================================================================
# Adaptation loop
# ============================================================================

def adapt_model(model, data_dir, adapt_iters, batch_size, block_size, 
                learning_rate, device, device_type, ctx, log_interval=50):
    """Run adaptation training loop."""
    optimizer = get_optimizer(model, learning_rate, weight_decay=0.01, device_type=device_type)
    
    model.train()
    losses = []
    
    for iter_num in range(adapt_iters):
        X, Y = get_batch(data_dir, 'train', batch_size, block_size, device, device_type)
        
        with ctx:
            logits, loss = model(X, Y)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        losses.append(loss.item())
        
        if iter_num % log_interval == 0:
            print(f"  iter {iter_num}: loss {loss.item():.4f}")
    
    return losses


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Continual Adaptation Experiment")
    parser.add_argument("--adapt_iters", type=int, default=500,
                        help="Number of adaptation iterations (default: 500)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for adaptation (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not specified)")
    parser.add_argument("--freeze_prototypes", action="store_true",
                        help="Freeze APN prototypes during adaptation")
    parser.add_argument("--eval_iters", type=int, default=50,
                        help="Iterations for loss estimation (default: 50)")
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
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # Setup dtype and autocast
    if device_type == 'cuda' and torch.cuda.is_bf16_supported():
        ptdtype = torch.bfloat16
    elif device_type == 'cuda':
        ptdtype = torch.float16
    else:
        ptdtype = torch.float32
    
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    print("=" * 60)
    print("Continual Adaptation Experiment: APN vs Baseline")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Adaptation iterations: {args.adapt_iters}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Prepare shifted dataset if needed
    shift_data_dir = "data/shakespeare_char_shift"
    if not os.path.exists(os.path.join(shift_data_dir, 'train.bin')):
        print("Preparing shifted dataset...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(shift_data_dir, 'prepare.py')], check=True)
        print()
    
    # Data directories
    original_data_dir = "data/shakespeare_char"
    
    # Check original data exists
    if not os.path.exists(os.path.join(original_data_dir, 'train.bin')):
        raise FileNotFoundError(
            f"Original dataset not found. Run: python {original_data_dir}/prepare.py"
        )
    
    # Load block_size from checkpoint
    baseline_ckpt_path = "out-shakespeare-char/ckpt.pt"
    apn_ckpt_path = "out-shakespeare-char-apn/ckpt.pt"
    
    if not os.path.exists(baseline_ckpt_path):
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {baseline_ckpt_path}\n"
            "Run: python train.py config/train_shakespeare_char.py"
        )
    
    if not os.path.exists(apn_ckpt_path):
        raise FileNotFoundError(
            f"APN checkpoint not found: {apn_ckpt_path}\n"
            "Run: python train.py config/train_shakespeare_char_apn.py"
        )
    
    results = {}
    
    # ========================================================================
    # Experiment 1: Baseline (MLP) - fine-tune all parameters
    # ========================================================================
    print("\n" + "=" * 60)
    print("BASELINE (MLP): Fine-tune all parameters")
    print("=" * 60)
    
    # Load baseline model
    model_baseline, model_args_baseline, _ = load_checkpoint("out-shakespeare-char", device)
    model_baseline.to(device)
    block_size = model_args_baseline['block_size']
    
    print(f"Loaded baseline model: {sum(p.numel() for p in model_baseline.parameters()):,} params")
    print(f"Block size: {block_size}")
    
    # Evaluate before adaptation
    print("\nEvaluating before adaptation...")
    baseline_pre_original = estimate_loss(
        model_baseline, original_data_dir, args.batch_size, block_size, 
        device, device_type, ctx, args.eval_iters
    )
    baseline_pre_shift = estimate_loss(
        model_baseline, shift_data_dir, args.batch_size, block_size,
        device, device_type, ctx, args.eval_iters
    )
    print(f"  Original domain val loss: {baseline_pre_original['val']:.4f}")
    print(f"  Shifted domain val loss: {baseline_pre_shift['val']:.4f}")
    
    # Adapt on shifted data (all parameters trainable)
    print(f"\nAdapting for {args.adapt_iters} iterations on shifted data...")
    trainable = sum(p.numel() for p in model_baseline.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,} (all)")
    
    start_time = time.time()
    adapt_model(
        model_baseline, shift_data_dir, args.adapt_iters, args.batch_size, block_size,
        args.learning_rate, device, device_type, ctx
    )
    baseline_adapt_time = time.time() - start_time
    
    # Evaluate after adaptation
    print("\nEvaluating after adaptation...")
    baseline_post_original = estimate_loss(
        model_baseline, original_data_dir, args.batch_size, block_size,
        device, device_type, ctx, args.eval_iters
    )
    baseline_post_shift = estimate_loss(
        model_baseline, shift_data_dir, args.batch_size, block_size,
        device, device_type, ctx, args.eval_iters
    )
    print(f"  Original domain val loss (retention): {baseline_post_original['val']:.4f}")
    print(f"  Shifted domain val loss (adaptation): {baseline_post_shift['val']:.4f}")
    
    results['baseline'] = {
        'name': 'Baseline (MLP)',
        'update_rule': 'Global fine-tune (all params)',
        'pre_original_val': baseline_pre_original['val'],
        'pre_shift_val': baseline_pre_shift['val'],
        'post_original_val': baseline_post_original['val'],
        'post_shift_val': baseline_post_shift['val'],
        'retention_ppl': math.exp(baseline_post_original['val']),
        'adaptation_ppl': math.exp(baseline_post_shift['val']),
        'adapt_time': baseline_adapt_time,
        'trainable_params': trainable
    }
    
    # Clean up
    del model_baseline
    if device_type == 'cuda':
        torch.cuda.empty_cache()
    
    # ========================================================================
    # Experiment 2: APN - update only active patch parameters
    # ========================================================================
    print("\n" + "=" * 60)
    print("APN: Update only patch parameters")
    print("=" * 60)
    
    # Load APN model
    model_apn, model_args_apn, _ = load_checkpoint("out-shakespeare-char-apn", device)
    model_apn.to(device)
    
    total_params = sum(p.numel() for p in model_apn.parameters())
    print(f"Loaded APN model: {total_params:,} params")
    print(f"APN config: K={model_args_apn.get('apn_K')}, k={model_args_apn.get('apn_k')}, r={model_args_apn.get('apn_r')}")
    
    # Evaluate before adaptation
    print("\nEvaluating before adaptation...")
    apn_pre_original = estimate_loss(
        model_apn, original_data_dir, args.batch_size, block_size,
        device, device_type, ctx, args.eval_iters
    )
    apn_pre_shift = estimate_loss(
        model_apn, shift_data_dir, args.batch_size, block_size,
        device, device_type, ctx, args.eval_iters
    )
    print(f"  Original domain val loss: {apn_pre_original['val']:.4f}")
    print(f"  Shifted domain val loss: {apn_pre_shift['val']:.4f}")
    
    # Freeze all except APN patches
    trainable_apn, frozen_apn = freeze_all_except_apn_patches(
        model_apn, freeze_prototypes=args.freeze_prototypes
    )
    print(f"\nFreezing non-patch parameters...")
    print(f"  Trainable: {trainable_apn:,} ({100*trainable_apn/total_params:.1f}%)")
    print(f"  Frozen: {frozen_apn:,} ({100*frozen_apn/total_params:.1f}%)")
    
    # List trainable parameters
    print("  Trainable parameter groups:")
    for name, param in model_apn.named_parameters():
        if param.requires_grad:
            print(f"    - {name}: {param.numel():,}")
    
    # Adapt on shifted data
    print(f"\nAdapting for {args.adapt_iters} iterations on shifted data...")
    
    start_time = time.time()
    adapt_model(
        model_apn, shift_data_dir, args.adapt_iters, args.batch_size, block_size,
        args.learning_rate, device, device_type, ctx
    )
    apn_adapt_time = time.time() - start_time
    
    # Evaluate after adaptation
    print("\nEvaluating after adaptation...")
    apn_post_original = estimate_loss(
        model_apn, original_data_dir, args.batch_size, block_size,
        device, device_type, ctx, args.eval_iters
    )
    apn_post_shift = estimate_loss(
        model_apn, shift_data_dir, args.batch_size, block_size,
        device, device_type, ctx, args.eval_iters
    )
    print(f"  Original domain val loss (retention): {apn_post_original['val']:.4f}")
    print(f"  Shifted domain val loss (adaptation): {apn_post_shift['val']:.4f}")
    
    results['apn'] = {
        'name': 'APN',
        'update_rule': 'Update only active patches (U, a, b, V' + (')' if args.freeze_prototypes else ', P)'),
        'pre_original_val': apn_pre_original['val'],
        'pre_shift_val': apn_pre_shift['val'],
        'post_original_val': apn_post_original['val'],
        'post_shift_val': apn_post_shift['val'],
        'retention_ppl': math.exp(apn_post_original['val']),
        'adaptation_ppl': math.exp(apn_post_shift['val']),
        'adapt_time': apn_adapt_time,
        'trainable_params': trainable_apn,
        'total_params': total_params
    }
    
    # ========================================================================
    # Generate output
    # ========================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Create output directory
    out_dir = "out-continual"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump({
            'config': {
                'adapt_iters': args.adapt_iters,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'device': device,
                'freeze_prototypes': args.freeze_prototypes
            },
            'results': results
        }, f, indent=2)
    print(f"Saved JSON: {json_path}")
    
    # Generate markdown
    md_content = generate_markdown(results, args)
    md_path = os.path.join(out_dir, "results.md")
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"Saved Markdown: {md_path}")
    
    # Print markdown
    print("\n" + md_content)
    
    return results


def generate_markdown(results, args):
    """Generate markdown results table matching the paper format."""
    lines = [
        "# Continual Learning Results: APN vs Baseline",
        "",
        "## Experiment Configuration",
        "",
        f"- **Adaptation iterations**: {args.adapt_iters}",
        f"- **Learning rate**: {args.learning_rate}",
        f"- **Batch size**: {args.batch_size}",
        f"- **Domain A**: shakespeare_char (original)",
        f"- **Domain B**: shakespeare_char_shift (synthetic characters)",
        "",
        "## Results",
        "",
        "| Model | Update Rule | Retention PPL | Adaptation PPL | Trainable Params |",
        "|-------|-------------|---------------|----------------|------------------|"
    ]
    
    for key in ['baseline', 'apn']:
        r = results[key]
        retention = f"{r['retention_ppl']:.2f}"
        adaptation = f"{r['adaptation_ppl']:.2f}"
        trainable = f"{r['trainable_params']:,}"
        lines.append(f"| {r['name']} | {r['update_rule']} | {retention} | {adaptation} | {trainable} |")
    
    # Add analysis
    baseline = results['baseline']
    apn = results['apn']
    
    retention_diff = apn['retention_ppl'] - baseline['retention_ppl']
    adaptation_diff = apn['adaptation_ppl'] - baseline['adaptation_ppl']
    
    lines.extend([
        "",
        "## Analysis",
        "",
        f"- **Retention** (lower is better): APN {'better' if retention_diff < 0 else 'worse'} by {abs(retention_diff):.2f} PPL",
        f"- **Adaptation** (lower is better): APN {'better' if adaptation_diff < 0 else 'worse'} by {abs(adaptation_diff):.2f} PPL",
        "",
        "### Interpretation",
        "",
        "- **Retention PPL**: Perplexity on original domain after adapting to new domain.",
        "  Lower = better preserved knowledge from pretraining.",
        "",
        "- **Adaptation PPL**: Perplexity on new domain after adaptation.",
        "  Lower = better learned the new patterns.",
        "",
        "### Key Insight",
        "",
        "APN's localized updates (only patch parameters) should reduce interference",
        "with the original domain while still allowing adaptation to the new domain.",
        f"In this experiment, APN updated only {apn['trainable_params']:,} parameters",
        f"({100*apn['trainable_params']/apn['total_params']:.1f}% of total)",
        f"while baseline updated all {baseline['trainable_params']:,} parameters."
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()
