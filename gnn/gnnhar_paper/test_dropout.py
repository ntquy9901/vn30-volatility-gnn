"""
Test dropout implementation in GNNHAR models.

Verifies that:
1. Dropout is active during training (model.train())
2. Dropout is disabled during evaluation (model.eval())
3. Dropout parameter is properly passed through create_model()

Run this after implementing dropout to verify correctness.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from gnn.gnnhar_paper.gnnhar_models import create_model


def test_dropout_behavior():
    """
    Test that dropout behaves correctly in train vs eval modes.

    With high dropout (0.5), multiple forward passes in train mode should
    produce different outputs (high std). In eval mode, outputs should be
    identical (zero std).
    """
    print("[TEST] Verifying dropout behavior...")

    # Create model with high dropout to make effect visible
    model = create_model('GNNHAR1L', n_hid=16, activation='relu', dropout=0.5)

    # Dummy data: 4 samples, 30 stocks, 3 HAR features
    X = torch.randn(4, 30, 3)
    # Identity adjacency matrix (no graph edges for this test)
    adj = torch.eye(30)

    # Test train mode (dropout should be active)
    model.train()
    preds_train = [model(X, adj).clone() for _ in range(10)]
    std_train = torch.stack(preds_train).std(dim=0).mean().item()
    print(f"  Train mode std: {std_train:.6f}")

    if std_train < 0.001:
        print(f"[ERROR] Dropout not active in train mode (std={std_train})")
        print("  Expected: std > 0.001 (random variation due to dropout)")
        return False
    else:
        print(f"  [OK] Dropout active in train mode (std={std_train:.6f})")

    # Test eval mode (dropout should be disabled)
    model.eval()
    preds_eval = [model(X, adj).clone() for _ in range(10)]
    std_eval = torch.stack(preds_eval).std(dim=0).mean().item()
    print(f"  Eval mode std:  {std_eval:.6f}")

    if std_eval > 1e-6:
        print(f"[ERROR] Dropout not disabled in eval mode (std={std_eval})")
        print("  Expected: std < 1e-6 (deterministic without dropout)")
        return False
    else:
        print(f"  [OK] Dropout disabled in eval mode (std={std_eval:.6f})")

    print("\n[OK] All dropout tests passed!")
    return True


def test_backward_compatibility():
    """
    Test that models work without dropout parameter (backward compatibility).
    """
    print("\n[TEST] Testing backward compatibility (dropout=0.0 default)...")

    # Create model without specifying dropout (should default to 0.0)
    model = create_model('GNNHAR1L', n_hid=16, activation='relu')

    # Verify dropout layer exists but rate is 0.0
    if hasattr(model, 'dropout'):
        print(f"  Dropout layer exists: {model.dropout}")
        print(f"  Dropout rate: {model.dropout_rate}")
        if model.dropout_rate == 0.0:
            print(f"  [OK] Default dropout rate is 0.0")
        else:
            print(f"  [ERROR] Expected dropout_rate=0.0, got {model.dropout_rate}")
            return False
    else:
        print(f"  [ERROR] Dropout layer not found")
        return False

    # Test that model works normally
    X = torch.randn(2, 30, 3)
    adj = torch.eye(30)
    output = model(X, adj)
    print(f"  Model output shape: {output.shape}")
    print(f"  [OK] Model works with default dropout=0.0")

    return True


def test_all_models():
    """
    Test that all GNNHAR models accept dropout parameter.
    """
    print("\n[TEST] Testing dropout support in all GNNHAR models...")

    models = ['GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L']
    X = torch.randn(2, 30, 3)
    adj = torch.eye(30)

    for model_name in models:
        try:
            model = create_model(model_name, n_hid=16, activation='relu', dropout=0.2)
            output = model(X, adj)
            print(f"  [OK] {model_name:12s} - output shape: {output.shape}")
        except Exception as e:
            print(f"  [ERROR] {model_name:12s} - {e}")
            return False

    # Test HAR (should not accept dropout)
    try:
        model = create_model('HAR')
        print(f"  [OK] HAR works (no dropout needed)")
    except Exception as e:
        print(f"  [ERROR] HAR - {e}")
        return False

    return True


if __name__ == "__main__":
    print("="*70)
    print("  DROPOUT IMPLEMENTATION TESTS")
    print("="*70 + "\n")

    # Run all tests
    all_passed = True
    all_passed &= test_dropout_behavior()
    all_passed &= test_backward_compatibility()
    all_passed &= test_all_models()

    print("\n" + "="*70)
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED - Please review errors above")
    print("="*70 + "\n")
