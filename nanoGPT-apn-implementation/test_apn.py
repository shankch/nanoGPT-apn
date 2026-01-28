"""
Unit tests for APN (Attractor Patch Network) implementation.
Verifies that:
1. Baseline MLP mode works correctly
2. APN mode works correctly
3. Both produce identical output shapes
4. Loss computation works for both
"""

import torch
from model import GPTConfig, GPT, APN, MLP, Block

def test_apn_implementation():
    print("=" * 60)
    print("APN Unit Tests")
    print("=" * 60)
    
    # Common test parameters
    batch_size = 2
    seq_len = 16
    n_embd = 64
    n_layer = 2
    n_head = 4
    block_size = 32
    vocab_size = 100
    
    # Create test input
    torch.manual_seed(42)
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # =========================================================
    # Test 1: Baseline GPTConfig(use_apn=False)
    # =========================================================
    print("\n[Test 1] Baseline MLP model (use_apn=False)")
    print("-" * 40)
    
    config_mlp = GPTConfig(
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        vocab_size=vocab_size,
        bias=True,
        dropout=0.1,
        use_apn=False,
    )
    
    model_mlp = GPT(config_mlp)
    model_mlp.eval()
    
    # Verify Block uses MLP
    assert isinstance(model_mlp.transformer.h[0].mlp, MLP), "Block should use MLP when use_apn=False"
    print("  ✓ Block uses MLP class")
    
    # Forward pass
    with torch.no_grad():
        logits_mlp, loss_mlp = model_mlp(idx, targets=targets)
    
    print(f"  ✓ Forward pass successful")
    print(f"    - Input shape: {idx.shape}")
    print(f"    - Logits shape: {logits_mlp.shape}")
    print(f"    - Loss: {loss_mlp.item():.4f}")
    
    # Verify shapes
    expected_logits_shape = (batch_size, seq_len, vocab_size)
    assert logits_mlp.shape == expected_logits_shape, f"Expected {expected_logits_shape}, got {logits_mlp.shape}"
    print(f"  ✓ Logits shape correct: {logits_mlp.shape}")
    
    # Verify loss is a scalar
    assert loss_mlp.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss_mlp), "Loss should not be NaN"
    assert not torch.isinf(loss_mlp), "Loss should not be Inf"
    print(f"  ✓ Loss is valid scalar")
    
    # =========================================================
    # Test 2: APN GPTConfig(use_apn=True)
    # =========================================================
    print("\n[Test 2] APN model (use_apn=True)")
    print("-" * 40)
    
    config_apn = GPTConfig(
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        vocab_size=vocab_size,
        bias=True,
        dropout=0.1,
        use_apn=True,
        apn_K=32,      # number of prototypes
        apn_k=4,       # top-k active patches
        apn_r=16,      # code dimension
        apn_tau=0.07,  # temperature
        apn_gamma=1.0, # residual scale
    )
    
    model_apn = GPT(config_apn)
    model_apn.eval()
    
    # Verify Block uses APN
    assert isinstance(model_apn.transformer.h[0].mlp, APN), "Block should use APN when use_apn=True"
    print("  ✓ Block uses APN class")
    
    # Forward pass
    with torch.no_grad():
        logits_apn, loss_apn = model_apn(idx, targets=targets)
    
    print(f"  ✓ Forward pass successful")
    print(f"    - Input shape: {idx.shape}")
    print(f"    - Logits shape: {logits_apn.shape}")
    print(f"    - Loss: {loss_apn.item():.4f}")
    
    # Verify shapes match baseline
    assert logits_apn.shape == logits_mlp.shape, f"Shape mismatch: APN {logits_apn.shape} vs MLP {logits_mlp.shape}"
    print(f"  ✓ Logits shape matches baseline: {logits_apn.shape}")
    
    # Verify loss is valid
    assert loss_apn.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss_apn), "Loss should not be NaN"
    assert not torch.isinf(loss_apn), "Loss should not be Inf"
    print(f"  ✓ Loss is valid scalar")
    
    # =========================================================
    # Test 3: APN module standalone
    # =========================================================
    print("\n[Test 3] APN module standalone")
    print("-" * 40)
    
    apn_module = APN(config_apn)
    x = torch.randn(batch_size, seq_len, n_embd)
    
    with torch.no_grad():
        y = apn_module(x)
    
    assert y.shape == x.shape, f"APN output shape {y.shape} != input shape {x.shape}"
    print(f"  ✓ APN input/output shapes match: {x.shape} -> {y.shape}")
    
    # =========================================================
    # Test 4: Block forward unchanged structure
    # =========================================================
    print("\n[Test 4] Block forward structure")
    print("-" * 40)
    
    block_mlp = Block(config_mlp)
    block_apn = Block(config_apn)
    
    x = torch.randn(batch_size, seq_len, n_embd)
    
    with torch.no_grad():
        y_mlp = block_mlp(x)
        y_apn = block_apn(x)
    
    assert y_mlp.shape == x.shape, f"MLP Block output shape mismatch"
    assert y_apn.shape == x.shape, f"APN Block output shape mismatch"
    print(f"  ✓ MLP Block: {x.shape} -> {y_mlp.shape}")
    print(f"  ✓ APN Block: {x.shape} -> {y_apn.shape}")
    
    # =========================================================
    # Test 5: Parameter counts
    # =========================================================
    print("\n[Test 5] Parameter counts")
    print("-" * 40)
    
    mlp_params = sum(p.numel() for p in model_mlp.parameters())
    apn_params = sum(p.numel() for p in model_apn.parameters())
    
    print(f"  MLP model parameters: {mlp_params:,}")
    print(f"  APN model parameters: {apn_params:,}")
    print(f"  Ratio (APN/MLP): {apn_params/mlp_params:.2f}x")
    
    # =========================================================
    # Test 6: Gradient flow
    # =========================================================
    print("\n[Test 6] Gradient flow")
    print("-" * 40)
    
    model_apn.train()
    logits, loss = model_apn(idx, targets=targets)
    loss.backward()
    
    # Check that APN parameters have gradients
    apn_block = model_apn.transformer.h[0].mlp
    grad_exists = all(p.grad is not None for p in apn_block.parameters())
    assert grad_exists, "APN parameters should have gradients"
    print("  ✓ All APN parameters have gradients")
    
    # Check specific APN components
    assert apn_block.prototypes.grad is not None, "Prototypes should have gradient"
    assert apn_block.V.grad is not None, "V should have gradient"
    assert apn_block.U.grad is not None, "U should have gradient"
    print("  ✓ Prototypes, V, U all have gradients")
    
    # =========================================================
    # Test 7: Inference mode (no targets)
    # =========================================================
    print("\n[Test 7] Inference mode")
    print("-" * 40)
    
    model_apn.eval()
    with torch.no_grad():
        logits_inf, loss_inf = model_apn(idx, targets=None)
    
    # In inference mode, only last position logits are returned
    assert logits_inf.shape == (batch_size, 1, vocab_size), f"Inference logits shape: {logits_inf.shape}"
    assert loss_inf is None, "Loss should be None when no targets provided"
    print(f"  ✓ Inference logits shape: {logits_inf.shape}")
    print(f"  ✓ Loss is None (as expected)")
    
    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - Baseline MLP model works correctly")
    print(f"  - APN model works correctly")
    print(f"  - Output shapes are identical")
    print(f"  - Loss computation works for both")
    print(f"  - Gradients flow through APN")
    
    return True

if __name__ == "__main__":
    test_apn_implementation()
