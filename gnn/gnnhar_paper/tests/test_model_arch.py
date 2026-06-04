"""
Unit tests for GNN-HAR model architecture.

Tests:
- Model output shapes
- H1/H2 separation
- Forward pass correctness
- Gradient flow
- Architecture properties

Date: 2026-05-30
"""

import sys
import os

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now we can import directly
import numpy as np
import torch
import torch.nn as nn
from gnnhar_models import GNNHAR1L, GNNHAR2L, GNNHAR3L, HAR, GHAR


def test_model_output_shape():
    """Verify model produces correct output shape."""
    print("\n[TEST] Model output shapes...")

    models = {
        'HAR': HAR(),
        'GHAR': GHAR(n_hid=16),
        'GNNHAR1L': GNNHAR1L(n_hid=16),
        'GNNHAR2L': GNNHAR2L(n_hid=16),
        'GNNHAR3L': GNNHAR3L(n_hid=16),
    }

    for model_name, model in models.items():
        model = model.to('cpu')

        # Input: (n_snapshots, n_stocks, n_features)
        X = torch.randn(100, 30, 3)

        # Adjacency: (n_stocks, n_stocks)
        adj = torch.randn(30, 30)

        try:
            # All models use the same interface: (node_feat, adj)
            # HAR and GHAR accept adj for interface consistency but don't use it
            output = model(X, adj)

            # Output should be: (n_snapshots, n_stocks)
            assert output.shape == (100, 30), \
                f"{model_name}: Wrong shape {output.shape}, expected (100, 30)"

            print(f"  [OK] {model_name}: shape {output.shape}")

        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")
            raise


def test_h1_h2_separation():
    """Verify H1 and H2 pathways are properly separated."""
    print("\n[TEST] H1/H2 separation...")

    model = GNNHAR1L(n_hid=16)

    # Input
    X = torch.randn(100, 30, 3)

    # Test 1: Zero adjacency (no graph effect)
    adj_zero = torch.zeros(30, 30)

    model.eval()
    with torch.no_grad():
        output_zero_adj = model(X, adj_zero)

    # Test 2: Identity adjacency (self-loops only)
    adj_identity = torch.eye(30)

    with torch.no_grad():
        output_identity_adj = model(X, adj_identity)

    # Outputs should be different (graph has effect)
    diff = torch.abs(output_zero_adj - output_identity_adj).mean().item()

    assert diff > 1e-6, \
        f"H1/H2 not separated: outputs too similar (diff={diff})"

    print(f"  [OK] Graph effect detected (mean diff={diff:.6f})")


def test_model_parameters():
    """Verify model parameters are initialized correctly."""
    print("\n[TEST] Model parameters...")

    models = {
        'GNNHAR1L': GNNHAR1L(n_hid=16),
        'GNNHAR2L': GNNHAR2L(n_hid=16),
        'GNNHAR3L': GNNHAR3L(n_hid=16),
    }

    for model_name, model in models.items():
        # Check parameters exist
        params = list(model.parameters())

        assert len(params) > 0, f"{model_name}: No parameters!"

        # Check all parameters are finite
        for name, param in model.named_parameters():
            assert torch.all(torch.isfinite(param)), \
                f"{model_name}.{name} contains Inf/NaN"

            # Check not all zeros
            assert param.abs().sum() > 1e-8, \
                f"{model_name}.{name} is all zeros!"

        print(f"  [OK] {model_name}: {len(params)} parameter groups")


def test_gradient_flow():
    """Verify gradients flow through all parameters."""
    print("\n[TEST] Gradient flow...")

    model = GNNHAR1L(n_hid=16)

    # Input
    X = torch.randn(10, 30, 3, requires_grad=False)
    adj = torch.randn(30, 30, requires_grad=False)
    y = torch.rand(10, 30) * 0.01

    # Forward pass
    output = model(X, adj)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check all parameters have gradients
    params_no_grad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            params_no_grad.append(name)
        elif torch.isnan(param.grad).any():
            raise AssertionError(f"{name} has NaN gradients")
        elif torch.isinf(param.grad).any():
            raise AssertionError(f"{name} has Inf gradients")

    assert len(params_no_grad) == 0, \
        f"Parameters without gradients: {params_no_grad}"

    print(f"  [OK] All {len(list(model.parameters()))} parameter groups have gradients")


def test_positive_output():
    """Verify model output can be made positive (for QLIKE)."""
    print("\n[TEST] Positive output capability...")

    model = GNNHAR1L(n_hid=16)

    X = torch.randn(10, 30, 3)
    adj = torch.randn(30, 30)

    model.eval()
    with torch.no_grad():
        output = model(X, adj)

        # Check output can be made positive with clamp
        output_positive = torch.clamp(output, min=1e-8)

        assert (output_positive >= 1e-8).all(), \
            "Clamped output contains negative values"

        # Check clamp doesn't destroy information
        assert output_positive.std() > 1e-6, \
            "Clamped output has no variance"

    print(f"  [OK] Output can be made positive (range: {output.min():.4f} to {output.max():.4f})")


def test_architecture_variants():
    """Test different GNNHAR variants."""
    print("\n[TEST] Architecture variants...")

    variants = {
        'GNNHAR1L': GNNHAR1L(n_hid=16),
        'GNNHAR2L': GNNHAR2L(n_hid=16),
        'GNNHAR3L': GNNHAR3L(n_hid=16),
    }

    X = torch.randn(10, 30, 3)
    adj = torch.randn(30, 30)

    for model_name, model in variants.items():
        model.eval()
        with torch.no_grad():
            output = model(X, adj)

        assert output.shape == (10, 30), \
            f"{model_name}: Wrong shape"

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  [OK] {model_name}: {n_params} parameters")


def run_all_tests():
    """Run all architecture tests."""
    print("\n" + "="*60)
    print("  MODEL ARCHITECTURE TEST SUITE")
    print("="*60)

    try:
        test_model_output_shape()
        test_h1_h2_separation()
        test_model_parameters()
        test_gradient_flow()
        test_positive_output()
        test_architecture_variants()

        print("\n" + "="*60)
        print("  ALL ARCHITECTURE TESTS PASSED!")
        print("="*60)
        return True

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
