#!/usr/bin/env python3
"""
Benchmark script comparing baseline MLP vs APN training on Shakespeare char dataset.

Runs both training configs and summarizes results including:
- Validation loss
- Parameter counts
- Training wall time
- MFU (if available in logs)

Usage:
    python bench_apn_vs_mlp.py [--max_iters N] [--device DEVICE]

Outputs:
    out-bench/summary.json
    out-bench/summary.md
"""

import os
import sys
import json
import time
import subprocess
import argparse
import re
from pathlib import Path

import torch


def run_training(config_path, extra_args=None, capture_output=True):
    """Run training and capture output, return wall time and stdout."""
    cmd = [sys.executable, "train.py", config_path]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    if capture_output:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) or "."
        )
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
        
        # Print output
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
    else:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)) or "."
        )
        stdout = ""
        stderr = ""
        returncode = result.returncode
    
    wall_time = time.time() - start_time
    
    if returncode != 0:
        print(f"WARNING: Training exited with code {returncode}")
    
    return {
        "wall_time": wall_time,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode
    }


def load_checkpoint(out_dir):
    """Load checkpoint and extract relevant fields."""
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"WARNING: Checkpoint not found at {ckpt_path}")
        return None
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    return {
        "best_val_loss": checkpoint.get("best_val_loss"),
        "iter_num": checkpoint.get("iter_num"),
        "model_args": checkpoint.get("model_args", {}),
        "config": checkpoint.get("config", {})
    }


def count_parameters(model_args):
    """Compute parameter count from model_args by instantiating the model."""
    try:
        from model import GPTConfig, GPT
        
        config = GPTConfig(**model_args)
        model = GPT(config)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Clean up
        del model
        
        return total_params
    except Exception as e:
        print(f"WARNING: Could not count parameters: {e}")
        return None


def parse_mfu_from_stdout(stdout):
    """Extract MFU values from training stdout."""
    # nanoGPT prints: "iter N: loss X.XXXX, time XXXms, mfu XX.XX%"
    mfu_pattern = r"mfu\s+([\d.]+)%"
    matches = re.findall(mfu_pattern, stdout)
    
    if matches:
        # Return the last MFU value (most representative of steady state)
        return float(matches[-1])
    return None


def parse_param_count_from_stdout(stdout):
    """Extract parameter count from model init output."""
    # nanoGPT prints: "number of parameters: X.XXM"
    pattern = r"number of parameters:\s*([\d.]+)M"
    match = re.search(pattern, stdout)
    
    if match:
        return float(match.group(1)) * 1e6
    return None


def prepare_data():
    """Ensure Shakespeare char data is prepared."""
    data_dir = "data/shakespeare_char"
    train_bin = os.path.join(data_dir, "train.bin")
    
    if not os.path.exists(train_bin):
        print("Preparing Shakespeare char dataset...")
        subprocess.run(
            [sys.executable, os.path.join(data_dir, "prepare.py")],
            check=True
        )
        print("Dataset prepared.\n")
    else:
        print("Dataset already prepared.\n")


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Benchmark APN vs MLP on Shakespeare char")
    parser.add_argument("--max_iters", type=int, default=None,
                        help="Override max_iters for faster benchmarking")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--compile", type=str, default=None,
                        help="Whether to use torch.compile (True/False)")
    parser.add_argument("--no_capture", action="store_true",
                        help="Don't capture output (show live)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("APN vs MLP BENCHMARK SCRIPT")
    print("=" * 70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Build extra args for training
    extra_args = []
    if args.max_iters is not None:
        extra_args.extend([f"--max_iters={args.max_iters}", 
                          f"--lr_decay_iters={args.max_iters}"])
    if args.device is not None:
        extra_args.append(f"--device={args.device}")
    if args.compile is not None:
        extra_args.append(f"--compile={args.compile}")
    
    # Ensure data is prepared
    prepare_data()
    
    # Define configs to benchmark
    configs = [
        {
            "name": "Baseline (MLP)",
            "config_path": "config/train_shakespeare_char.py",
            "out_dir": "out-shakespeare-char"
        },
        {
            "name": "APN",
            "config_path": "config/train_shakespeare_char_apn.py", 
            "out_dir": "out-shakespeare-char-apn"
        }
    ]
    
    results = []
    
    print("\n" + "="*60)
    print("BENCHMARK: APN vs MLP on Shakespeare Char")
    print("="*60)
    
    for cfg in configs:
        print(f"\n>>> Training {cfg['name']}...")
        
        # Run training
        train_result = run_training(
            cfg["config_path"], 
            extra_args=extra_args,
            capture_output=not args.no_capture
        )
        
        # Load checkpoint
        ckpt_data = load_checkpoint(cfg["out_dir"])
        
        # Parse stdout for additional info
        mfu = parse_mfu_from_stdout(train_result["stdout"])
        param_count_stdout = parse_param_count_from_stdout(train_result["stdout"])
        
        # Compute parameters programmatically if we have model_args
        param_count_computed = None
        if ckpt_data and ckpt_data["model_args"]:
            param_count_computed = count_parameters(ckpt_data["model_args"])
        
        result = {
            "name": cfg["name"],
            "config_path": cfg["config_path"],
            "out_dir": cfg["out_dir"],
            "wall_time_seconds": train_result["wall_time"],
            "wall_time_formatted": format_time(train_result["wall_time"]),
            "returncode": train_result["returncode"],
            "best_val_loss": ckpt_data["best_val_loss"] if ckpt_data else None,
            "iter_num": ckpt_data["iter_num"] if ckpt_data else None,
            "model_args": ckpt_data["model_args"] if ckpt_data else None,
            "config": ckpt_data["config"] if ckpt_data else None,
            "param_count": param_count_computed or param_count_stdout,
            "mfu_percent": mfu
        }
        
        results.append(result)
        
        print(f"\n{cfg['name']} completed:")
        print(f"  Val Loss: {result['best_val_loss']:.4f}" if result['best_val_loss'] else "  Val Loss: N/A")
        print(f"  Iterations: {result['iter_num']}")
        print(f"  Wall Time: {result['wall_time_formatted']}")
        if result['param_count']:
            print(f"  Parameters: {result['param_count']/1e6:.2f}M")
        if result['mfu_percent']:
            print(f"  MFU: {result['mfu_percent']:.2f}%")
    
    # Create output directory
    out_bench_dir = "out-bench"
    os.makedirs(out_bench_dir, exist_ok=True)
    
    # Save JSON summary
    json_path = os.path.join(out_bench_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "benchmark": "APN vs MLP Shakespeare Char",
            "extra_args": extra_args,
            "results": results
        }, f, indent=2, default=str)
    print(f"\nSaved JSON summary to {json_path}")
    
    # Generate markdown summary
    md_content = generate_markdown_summary(results, extra_args)
    md_path = os.path.join(out_bench_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Saved Markdown summary to {md_path}")
    
    # Print summary table
    print("\n" + md_content)
    
    return results


def generate_markdown_summary(results, extra_args):
    """Generate markdown summary table."""
    lines = [
        "# APN vs MLP Benchmark Results",
        "",
        "## Configuration",
        "",
        f"- Dataset: Shakespeare Char (65 vocab, ~1M tokens)",
        f"- Extra args: {' '.join(extra_args) if extra_args else 'None (default configs)'}",
        "",
        "## Results",
        "",
        "| Model | Val Loss | Params | Wall Time | MFU | Iterations |",
        "|-------|----------|--------|-----------|-----|------------|"
    ]
    
    for r in results:
        val_loss = f"{r['best_val_loss']:.4f}" if r['best_val_loss'] else "N/A"
        params = f"{r['param_count']/1e6:.2f}M" if r['param_count'] else "N/A"
        wall_time = r['wall_time_formatted']
        mfu = f"{r['mfu_percent']:.2f}%" if r['mfu_percent'] else "N/A"
        iters = str(r['iter_num']) if r['iter_num'] else "N/A"
        
        lines.append(f"| {r['name']} | {val_loss} | {params} | {wall_time} | {mfu} | {iters} |")
    
    # Add comparison
    if len(results) == 2 and all(r['best_val_loss'] for r in results):
        baseline = results[0]
        apn = results[1]
        
        loss_diff = apn['best_val_loss'] - baseline['best_val_loss']
        loss_pct = (loss_diff / baseline['best_val_loss']) * 100
        
        lines.extend([
            "",
            "## Comparison",
            "",
            f"- Loss difference (APN - Baseline): {loss_diff:+.4f} ({loss_pct:+.2f}%)"
        ])
        
        if baseline['param_count'] and apn['param_count']:
            param_ratio = apn['param_count'] / baseline['param_count']
            lines.append(f"- Parameter ratio (APN / Baseline): {param_ratio:.2f}x")
        
        if baseline['wall_time_seconds'] and apn['wall_time_seconds']:
            time_ratio = apn['wall_time_seconds'] / baseline['wall_time_seconds']
            lines.append(f"- Wall time ratio (APN / Baseline): {time_ratio:.2f}x")
    
    # Add APN config details
    apn_result = next((r for r in results if "APN" in r['name']), None)
    if apn_result and apn_result.get('model_args'):
        ma = apn_result['model_args']
        if ma.get('use_apn'):
            lines.extend([
                "",
                "## APN Configuration",
                "",
                f"- `apn_K` (prototypes): {ma.get('apn_K', 'N/A')}",
                f"- `apn_k` (top-k): {ma.get('apn_k', 'N/A')}",
                f"- `apn_r` (rank): {ma.get('apn_r', 'N/A')}",
                f"- `apn_tau` (temperature): {ma.get('apn_tau', 'N/A')}",
                f"- `apn_gamma` (residual scale): {ma.get('apn_gamma', 'N/A')}"
            ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()
