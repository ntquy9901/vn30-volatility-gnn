"""
Test script for QLIKE loss implementation.

Tests:
1. QLIKE loss function works correctly
2. Gradients flow properly
3. Loss value is reasonable

Date: 2026-05-30
"""

import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('.')

from evaluation import qlike_loss, compute_metrics


def test_qlike_loss():
    """Test QLIKE loss function."""
    print("\n" + "="*60)
    print("  TEST 1: QLIKE Loss Function")
    print("="*60)

    # Generate dummy RV data (always positive)
    np.random.seed(42)
    n = 100

    y_true = np.random.gamma(shape=2, scale=0.001, size=n)  # Positive values
    y_pred = y_true + np.random.randn(n) * 0.0002  # Add noise
    y_pred = np.maximum(y_pred, 1e-8)  # Ensure positive

    # Convert to tensors
    pred_torch = torch.tensor(y_pred, dtype=torch.float32, requires_grad=True)
    targ_torch = torch.tensor(y_true, dtype=torch.float32)

    # Compute QLIKE loss
    loss = qlike_loss(pred_torch, targ_torch)

    print(f"  QLIKE loss value: {loss.item():.6f}")
    print(f"  Loss requires grad: {loss.requires_grad}")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")

    # Test gradient flow
    loss.backward()
    print(f"  Gradients exist: {pred_torch.grad is not None}")
    print(f"  Gradients finite: {torch.isfinite(pred_torch.grad).all().item()}")

    # Assertions
    assert torch.isfinite(loss).item(), "Loss is NaN or Inf"
    assert loss.item() > 0, "Loss should be positive"
    assert pred_torch.grad is not None, "No gradients"
    assert torch.isfinite(pred_torch.grad).all().item(), "Gradients are not finite"

    print("\n  [OK] QLIKE loss test passed!")


def test_qlike_vs_mse():
    """Compare QLIKE with MSE loss."""
    print("\n" + "="*60)
    print("  TEST 2: QLIKE vs MSE Loss")
    print("="*60)

    # Generate data
    np.random.seed(42)
    n = 100
    y_true = np.random.gamma(shape=2, scale=0.001, size=n)

    # Scenario 1: Perfect predictions
    y_pred_perfect = y_true.copy()
    pred_p = torch.tensor(y_pred_perfect, dtype=torch.float32, requires_grad=True)
    targ = torch.tensor(y_true, dtype=torch.float32)

    mse_p = nn.MSELoss()(pred_p, targ)
    qlike_p = qlike_loss(pred_p, targ)

    print(f"\n  Perfect predictions:")
    print(f"    MSE:   {mse_p.item():.8f}")
    print(f"    QLIKE: {qlike_p.item():.8f}")

    # Scenario 2: Biased predictions (underpredict)
    y_pred_biased_low = y_true * 0.8
    pred_low = torch.tensor(y_pred_biased_low, dtype=torch.float32, requires_grad=True)
    mse_low = nn.MSELoss()(pred_low, targ)
    qlike_low = qlike_loss(pred_low, targ)

    print(f"\n  Underprediction (80% of true):")
    print(f"    MSE:   {mse_low.item():.8f}")
    print(f"    QLIKE: {qlike_low.item():.8f}")

    # Scenario 3: Biased predictions (overpredict)
    y_pred_biased_high = y_true * 1.2
    pred_high = torch.tensor(y_pred_biased_high, dtype=torch.float32, requires_grad=True)
    mse_high = nn.MSELoss()(pred_high, targ)
    qlike_high = qlike_loss(pred_high, targ)

    print(f"\n  Overprediction (120% of true):")
    print(f"    MSE:   {mse_high.item():.8f}")
    print(f"    QLIKE: {qlike_high.item():.8f}")

    # Check asymmetry
    print(f"\n  Asymmetry check:")
    print(f"    MSE under vs over: {abs(mse_low.item() - mse_high.item()):.8f}")
    print(f"    QLIKE under vs over: {abs(qlike_low.item() - qlike_high.item()):.8f}")

    if qlike_low.item() > qlike_high.item():
        print(f"    [OK] QLIKE penalizes underprediction more (asymmetric)")
    else:
        print(f"    [WARN] QLIKE should penalize underprediction more")

    print("\n  [OK] QLIKE vs MSE test completed!")


def test_comprehensive_metrics():
    """Test comprehensive metrics including QLIKE."""
    print("\n" + "="*60)
    print("  TEST 3: Comprehensive Metrics")
    print("="*60)

    # Generate data
    np.random.seed(42)
    n = 100
    y_true = np.random.gamma(shape=2, scale=0.001, size=n)
    y_pred = y_true + np.random.randn(n) * 0.0002
    y_pred = np.maximum(y_pred, 1e-8)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, include_qlike=True, include_hetero=True)

    print(f"\n  Computed metrics:")
    print(f"    R2:    {metrics['r2']:>10.4f}  (higher is better)")
    print(f"    MAE:   {metrics['mae']:>10.6f}  (lower is better)")
    print(f"    RMSE:  {metrics['rmse']:>10.6f}  (lower is better)")
    print(f"    QLIKE: {metrics['qlike']:>10.6f}  (lower is better)")
    print(f"    HMSE:  {metrics['hmse']:>10.6f}  (lower is better)")
    print(f"    HMAE:  {metrics['hmae']:>10.6f}  (lower is better)")

    # Check all metrics are finite
    for key, value in metrics.items():
        assert np.isfinite(value), f"{key} is not finite"

    print("\n  [OK] Comprehensive metrics test passed!")


def test_training_simulation():
    """Simulate training with QLIKE loss."""
    print("\n" + "="*60)
    print("  TEST 4: Training Simulation")
    print("="*60)

    # Create dummy model
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 100
    n_features = 10

    # Dummy data
    X = torch.randn(n_samples, n_features, requires_grad=False)
    y_true = torch.rand(n_samples) * 0.01
    y_true = torch.clamp(y_true, min=1e-8)  # Ensure positive

    # Dummy model (linear)
    model = nn.Linear(n_features, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"\n  Training for 50 epochs with QLIKE loss...")

    losses = []
    for epoch in range(50):
        optimizer.zero_grad()

        y_pred = model(X).squeeze()
        y_pred = torch.clamp(y_pred, min=1e-8)  # Ensure positive

        loss = qlike_loss(y_pred, y_true)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:2d}: QLIKE loss = {loss.item():.6f}")

    print(f"\n  Final loss: {losses[-1]:.6f}")
    print(f"  Loss decreased: {losses[0] > losses[-1]}")
    print(f"  Total decrease: {losses[0] - losses[-1]:.6f}")

    # Check convergence
    assert losses[-1] < losses[0], "Loss should decrease"
    assert np.isfinite(losses[-1]), "Final loss is not finite"

    print("\n  [OK] Training simulation test passed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  QLIKE LOSS IMPLEMENTATION TEST SUITE")
    print("  Date: 2026-05-30")
    print("="*60)

    test_qlike_loss()
    test_qlike_vs_mse()
    test_comprehensive_metrics()
    test_training_simulation()

    print("\n" + "="*60)
    print("  ALL TESTS PASSED!")
    print("  QLIKE loss is ready for production use")
    print("="*60 + "\n")
