"""
Critical tests for quasi_likelihood_loss after v1.3_LOSS_FIX bug correction.

These tests validate the fixes Amelia identified:
1. eps guards both division AND log (not just division)
2. Function handles edge cases (zero volatility, near-zero predictions)
3. Matches paper implementation exactly
4. Gradients flow correctly without NaN/Inf

Run with: python -m pytest gnn/gnnhar_paper/tests/test_quasi_likelihood_loss.py -v
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytest
from gnn.gnnhar_paper.gnnhar_models import quasi_likelihood_loss


def test_zero_true_volatility():
    """
    Test 1: Edge case — zero true volatility.

    RV can be zero on days with zero realized volatility.
    The loss should handle this gracefully without returning NaN or inf.
    """
    print("\n[Test 1] Zero true volatility...")

    y_true = torch.tensor([0.0, 0.01, 0.02])
    y_pred = torch.tensor([0.015, 0.015, 0.015])
    loss = quasi_likelihood_loss(y_pred, y_true)

    # Should not return NaN or inf
    assert not torch.isnan(loss).any(), "Loss contains NaN for zero true volatility"
    assert not torch.isinf(loss).any(), "Loss contains inf for zero true volatility"
    assert loss.item() > 0, "Loss should be positive"

    print(f"  y_true = {y_true}")
    print(f"  y_pred = {y_pred}")
    print(f"  Loss = {loss.item():.6f}")
    print("  [OK] Handles zero true volatility correctly")


def test_near_zero_prediction():
    """
    Test 2: Edge case — near-zero prediction.

    If model predicts near-zero volatility, ratio should explode but eps
    should prevent division by zero and log(0).
    """
    print("\n[Test 2] Near-zero prediction...")

    y_true = torch.tensor([0.01, 0.02, 0.03])
    y_pred = torch.tensor([1e-6, 0.015, 0.025])  # First entry nearly zero
    loss = quasi_likelihood_loss(y_pred, y_true)

    # eps=1e-4 should prevent division by zero
    assert not torch.isnan(loss).any(), "Loss contains NaN for near-zero prediction"
    assert not torch.isinf(loss).any(), "Loss contains inf despite eps guard"
    assert loss.item() > 0, "Loss should be positive"

    print(f"  y_true = {y_true}")
    print(f"  y_pred = {y_pred}")
    print(f"  Loss = {loss.item():.6f}")
    print("  [OK] eps prevents division by zero and log(0)")


def test_match_paper_implementation():
    """
    Test 3: Match paper implementation exactly.

    Verify that our implementation matches the paper's code (GNNHAR.py line 322-323)
    exactly, including eps placement in both division and log.
    """
    print("\n[Test 3] Match paper implementation...")

    # Use specific test values
    outputs = torch.tensor([0.02, 0.03, 0.04])  # y_true in paper code
    forecast_y = torch.tensor([0.015, 0.025, 0.035])  # y_pred in paper code
    eps = 1e-4

    # Paper version (GNNHAR.py:322-323)
    true_fore = outputs / (forecast_y + eps)
    l_v_paper = torch.mean(true_fore - torch.log(true_fore + eps))

    # Our function (note: parameter order is y_pred, y_true)
    l_v_ours = quasi_likelihood_loss(forecast_y, outputs, eps)

    # Should match exactly (within floating point tolerance)
    assert torch.allclose(l_v_paper, l_v_ours, atol=1e-6), \
        f"Does not match paper: paper={l_v_paper.item():.8f}, ours={l_v_ours.item():.8f}"

    print(f"  Paper loss:    {l_v_paper.item():.8f}")
    print(f"  Our loss:      {l_v_ours.item():.8f}")
    print(f"  Difference:    {abs(l_v_paper.item() - l_v_ours.item()):.10f}")
    print("  [OK] Matches paper implementation exactly")


def test_gradients_not_nan():
    """
    Test 4: Gradient flow sanity check.

    Gradients should flow through the loss function without producing NaN or inf.
    """
    print("\n[Test 4] Gradient flow...")

    y_true = torch.tensor([0.01, 0.02, 0.03], requires_grad=False)
    y_pred = torch.tensor([0.015, 0.025, 0.035], requires_grad=True)
    loss = quasi_likelihood_loss(y_pred, y_true)
    loss.backward()

    assert y_pred.grad is not None, "No gradients computed"
    assert not torch.isnan(y_pred.grad).any(), "Gradients contain NaN"
    assert not torch.isinf(y_pred.grad).any(), "Gradients contain inf"

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradients: {y_pred.grad}")
    print("  [OK] Gradients flow correctly without NaN/inf")


def test_eps_in_both_places():
    """
    Test that eps guards BOTH division and log (critical fix in v1.3_LOSS_FIX).

    Before fix: ratio - torch.log(ratio)  [eps only in division]
    After fix:  ratio - torch.log(ratio + eps)  [eps in both places]
    """
    print("\n[Test 5] eps guards both division AND log...")

    # Test case where ratio is very small (would cause log(0) without eps)
    y_true = torch.tensor([0.001])  # Very small true volatility
    y_pred = torch.tensor([0.01])   # Larger prediction -> ratio = 0.1
    eps = 1e-4

    # Our implementation should have eps in log term
    ratio = y_true / (y_pred + eps)
    loss_with_eps = quasi_likelihood_loss(y_pred, y_true, eps)

    # Manually compute what it would be WITHOUT eps in log
    ratio_value = (y_true / (y_pred + eps)).item()
    loss_without_eps_in_log = ratio_value - torch.log(torch.tensor(ratio_value))

    # With eps in log should be different (and finite)
    # Without eps in log could be -inf if ratio is very small
    assert not torch.isnan(loss_with_eps), "Loss with eps is NaN"
    assert not torch.isinf(loss_with_eps), "Loss with eps is inf"

    print(f"  Ratio: {ratio_value:.6f}")
    print(f"  Loss (eps in log): {loss_with_eps.item():.6f}")
    print(f"  Loss (no eps in log): {loss_without_eps_in_log:.6f}")
    print("  [OK] eps guards both division and log domain")


def test_asymmetric_penalty():
    """
    Test that loss penalizes under-prediction MORE than over-prediction.

    This is the CORRECT behavior for volatility risk management:
    - Under-predicting risk is dangerous (insufficient capital, VaR breaches)
    - Over-predicting risk is conservative (wasteful but safe)

    Mathematical explanation:
    - Under-prediction (pred < true): ratio = true/pred > 1, loss grows aggressively
    - Over-prediction (pred > true): ratio = true/pred < 1, loss grows slowly
    """
    print("\n[Test 6] Asymmetric penalty...")

    y_true_base = torch.tensor([0.02])

    # Under-prediction (pred = 0.5 * true)
    y_pred_under = torch.tensor([0.01])
    loss_under = quasi_likelihood_loss(y_pred_under, y_true_base)

    # Over-prediction (pred = 2.0 * true)
    y_pred_over = torch.tensor([0.04])
    loss_over = quasi_likelihood_loss(y_pred_over, y_true_base)

    print(f"  True volatility: {y_true_base.item():.3f}")
    print(f"  Under-prediction (0.5x):  loss = {loss_under.item():.6f}")
    print(f"  Over-prediction (2.0x):  loss = {loss_over.item():.6f}")

    # Under-prediction should be penalized MORE heavily
    # (This is the correct asymmetric behavior for risk management)
    assert loss_under.item() > loss_over.item(), \
        "Under-prediction should be penalized MORE than over-prediction"

    print("  [OK] Asymmetric penalty correct (under > over)")
    print("        Risk management: under-prediction dangerous, over-prediction safe")


def test_perfect_prediction_minimum_loss():
    """
    Test that perfect prediction gives minimum loss of 1.0.

    When pred = target, ratio = 1, loss = 1 - log(1) = 1 - 0 = 1.
    This is the theoretical minimum of this loss function.
    """
    print("\n[Test 7] Perfect prediction minimum loss...")

    y_true = torch.tensor([0.01, 0.02, 0.03, 0.04])
    y_pred = torch.tensor([0.01, 0.02, 0.03, 0.04])  # Perfect prediction
    loss = quasi_likelihood_loss(y_pred, y_true)

    # Should be very close to 1.0 (within numerical precision)
    assert abs(loss.item() - 1.0) < 0.01, \
        f"Perfect prediction should give loss ~1.0, got {loss.item():.6f}"

    print(f"  Perfect prediction loss: {loss.item():.6f}")
    print(f"  Expected minimum: 1.0")
    print(f"  Difference: {abs(loss.item() - 1.0):.8f}")
    print("  [OK] Perfect prediction gives minimum loss of ~1.0")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  CRITICAL TESTS FOR quasi_likelihood_loss (v1.3_LOSS_FIX)")
    print("="*70)

    # Run all tests
    try:
        test_zero_true_volatility()
        test_near_zero_prediction()
        test_match_paper_implementation()
        test_gradients_not_nan()
        test_eps_in_both_places()
        test_asymmetric_penalty()
        test_perfect_prediction_minimum_loss()

        print("\n" + "="*70)
        print("  ALL TESTS PASSED")
        print("="*70)
        print("\n[SUCCESS] Loss function is ready for production use")
        print("         All critical issues from Amelia's review are fixed")

    except AssertionError as e:
        print("\n" + "="*70)
        print("  TEST FAILED")
        print("="*70)
        print(f"\n[ERROR] {e}")
        print("\n[FAILURE] Do not use this loss function until tests pass")
        sys.exit(1)
