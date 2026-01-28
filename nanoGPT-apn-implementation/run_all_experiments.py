#!/usr/bin/env python3
"""
Master script to run all APN experiments in order.

This script will:
1. Prepare the Shakespeare char dataset
2. Prepare the shifted domain dataset  
3. Train baseline MLP model
4. Train APN model
5. Run continual learning experiment
6. Generate samples from both models
7. Print final summary

Usage:
    python run_all_experiments.py

For cloud GPU:
    python run_all_experiments.py --device=cuda
"""

import os
import sys
import time
import subprocess
import argparse
import json


def print_header(msg):
    """Print a prominent header."""
    print()
    print("=" * 70)
    print(f"  {msg}")
    print("=" * 70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_step(step_num, total, msg):
    """Print step progress."""
    print()
    print(f"[STEP {step_num}/{total}] {msg}")
    print("-" * 50)


def run_command(cmd, description, check=True):
    """Run a command and print status."""
    print(f">>> Running: {' '.join(cmd)}")
    print()
    start = time.time()
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print()
        print(f"✓ {description} completed in {elapsed:.1f}s")
    else:
        print()
        print(f"✗ {description} FAILED (exit code {result.returncode})")
        if check:
            sys.exit(1)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all APN experiments")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda, cpu, mps). Default: cuda")
    parser.add_argument("--max_iters", type=int, default=5000,
                        help="Max training iterations. Default: 5000")
    parser.add_argument("--adapt_iters", type=int, default=500,
                        help="Continual adaptation iterations. Default: 500")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training if checkpoints exist")
    args = parser.parse_args()
    
    total_steps = 7
    results = {}
    
    print_header("APN EXPERIMENTS - MASTER RUNNER")
    print(f"Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Adaptation iterations: {args.adapt_iters}")
    print(f"  Skip training: {args.skip_training}")
    
    overall_start = time.time()
    
    # =========================================================================
    # STEP 1: Prepare Shakespeare char dataset
    # =========================================================================
    print_step(1, total_steps, "Prepare Shakespeare char dataset")
    
    if os.path.exists("data/shakespeare_char/train.bin"):
        print("Dataset already exists, skipping...")
    else:
        run_command(
            [sys.executable, "data/shakespeare_char/prepare.py"],
            "Shakespeare char dataset preparation"
        )
    
    # =========================================================================
    # STEP 2: Prepare shifted domain dataset
    # =========================================================================
    print_step(2, total_steps, "Prepare shifted domain dataset (Domain B)")
    
    if os.path.exists("data/shakespeare_char_shift/train.bin"):
        print("Shifted dataset already exists, skipping...")
    else:
        run_command(
            [sys.executable, "data/shakespeare_char_shift/prepare.py"],
            "Shifted dataset preparation"
        )
    
    # =========================================================================
    # STEP 3: Train baseline MLP model
    # =========================================================================
    print_step(3, total_steps, "Train BASELINE (MLP) model")
    
    baseline_exists = os.path.exists("out-shakespeare-char/ckpt.pt")
    
    if baseline_exists and args.skip_training:
        print("Baseline checkpoint exists, skipping training...")
    else:
        train_start = time.time()
        run_command(
            [sys.executable, "train.py", "config/train_shakespeare_char.py",
             f"--device={args.device}",
             f"--max_iters={args.max_iters}",
             f"--lr_decay_iters={args.max_iters}"],
            "Baseline MLP training"
        )
        results['baseline_train_time'] = time.time() - train_start
    
    # =========================================================================
    # STEP 4: Train APN model
    # =========================================================================
    print_step(4, total_steps, "Train APN model")
    
    apn_exists = os.path.exists("out-shakespeare-char-apn/ckpt.pt")
    
    if apn_exists and args.skip_training:
        print("APN checkpoint exists, skipping training...")
    else:
        train_start = time.time()
        run_command(
            [sys.executable, "train.py", "config/train_shakespeare_char_apn.py",
             f"--device={args.device}",
             f"--max_iters={args.max_iters}",
             f"--lr_decay_iters={args.max_iters}"],
            "APN training"
        )
        results['apn_train_time'] = time.time() - train_start
    
    # =========================================================================
    # STEP 5: Run continual learning experiment
    # =========================================================================
    print_step(5, total_steps, "Run continual learning experiment")
    
    run_command(
        [sys.executable, "continual_apn.py",
         f"--device={args.device}",
         f"--adapt_iters={args.adapt_iters}"],
        "Continual learning experiment"
    )
    
    # =========================================================================
    # STEP 6: Generate samples
    # =========================================================================
    print_step(6, total_steps, "Generate text samples from both models")
    
    run_command(
        [sys.executable, "generate_samples.py",
         f"--device={args.device}",
         "--num_samples=3",
         "--max_new_tokens=500"],
        "Sample generation"
    )
    
    # =========================================================================
    # STEP 7: Print final summary
    # =========================================================================
    print_step(7, total_steps, "Final Summary")
    
    overall_time = time.time() - overall_start
    
    print_header("EXPERIMENT RESULTS SUMMARY")
    
    # Load and print benchmark results if available
    if os.path.exists("out-bench/summary.json"):
        with open("out-bench/summary.json") as f:
            bench_data = json.load(f)
        print("=== LANGUAGE MODELING RESULTS ===")
        print()
        for r in bench_data.get('results', []):
            print(f"{r['name']}:")
            print(f"  Val Loss: {r.get('best_val_loss', 'N/A')}")
            if r.get('best_val_loss'):
                import math
                print(f"  Val PPL:  {math.exp(r['best_val_loss']):.2f}")
            print(f"  Params:   {r.get('param_count', 'N/A')}")
            print()
    
    # Load and print continual learning results
    if os.path.exists("out-continual/results.json"):
        with open("out-continual/results.json") as f:
            cl_data = json.load(f)
        print("=== CONTINUAL LEARNING RESULTS ===")
        print()
        for name, r in cl_data.get('results', {}).items():
            print(f"{r.get('name', name)}:")
            print(f"  Retention PPL: {r.get('retention_ppl', 'N/A'):.2f}" if r.get('retention_ppl') else f"  Retention PPL: N/A")
            print(f"  Adaptation PPL: {r.get('adaptation_ppl', 'N/A'):.2f}" if r.get('adaptation_ppl') else f"  Adaptation PPL: N/A")
            print(f"  Trainable params: {r.get('trainable_params', 'N/A'):,}" if r.get('trainable_params') else f"  Trainable params: N/A")
            print()
    
    # Print sample previews
    print("=== SAMPLE PREVIEWS ===")
    print()
    
    for model_name, out_dir in [("Baseline", "out-shakespeare-char"), ("APN", "out-shakespeare-char-apn")]:
        sample_file = f"{out_dir}/samples/baseline_01.txt" if model_name == "Baseline" else f"{out_dir}/samples/apn_01.txt"
        if os.path.exists(sample_file):
            with open(sample_file) as f:
                sample = f.read()[:300]
            print(f"{model_name} model sample (first 300 chars):")
            print("-" * 40)
            print(sample)
            print("-" * 40)
            print()
    
    print("=== TIMING ===")
    print(f"Total experiment time: {overall_time/60:.1f} minutes")
    if 'baseline_train_time' in results:
        print(f"Baseline training time: {results['baseline_train_time']/60:.1f} minutes")
    if 'apn_train_time' in results:
        print(f"APN training time: {results['apn_train_time']/60:.1f} minutes")
    
    print()
    print("=== OUTPUT FILES ===")
    print("Checkpoints:")
    print("  out-shakespeare-char/ckpt.pt")
    print("  out-shakespeare-char-apn/ckpt.pt")
    print("Results:")
    print("  out-continual/results.json")
    print("  out-continual/results.md")
    print("Samples:")
    print("  out-shakespeare-char/samples/")
    print("  out-shakespeare-char-apn/samples/")
    
    print()
    print_header("ALL EXPERIMENTS COMPLETED!")
    
    # Print the markdown results for easy copy-paste
    if os.path.exists("out-continual/results.md"):
        print()
        print("=== MARKDOWN RESULTS (copy this for the paper) ===")
        print()
        with open("out-continual/results.md") as f:
            print(f.read())


if __name__ == "__main__":
    main()
