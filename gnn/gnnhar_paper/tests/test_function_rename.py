"""
Test that the function rename works correctly and deprecation warning is shown.

This validates Paige's recommendations:
1. gnnhar_ratio_loss works correctly
2. quasi_likelihood_loss still works but shows deprecation warning
3. Both functions produce identical results
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import warnings
from gnn.gnnhar_paper.gnnhar_models import gnnhar_ratio_loss, quasi_likelihood_loss

print("\n" + "="*70)
print("  FUNCTION RENAME TEST")
print("="*70)

# Test data
y_pred = torch.tensor([0.015, 0.025, 0.035])
y_true = torch.tensor([0.02, 0.03, 0.04])

print("\n[Test 1] gnnhar_ratio_loss works correctly...")
loss_new = gnnhar_ratio_loss(y_pred, y_true)
print(f"  gnnhar_ratio_loss result: {loss_new.item():.6f}")
assert not torch.isnan(loss_new), "gnnhar_ratio_loss produced NaN"
assert loss_new.item() > 0, "gnnhar_ratio_loss should be positive"
print("  [OK] gnnhar_ratio_loss works")

print("\n[Test 2] quasi_likelihood_loss shows deprecation warning...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    loss_old = quasi_likelihood_loss(y_pred, y_true)

    # Check that deprecation warning was raised
    assert len(w) >= 1, "No warnings raised"
    assert any(issubclass(warning.category, DeprecationWarning) for warning in w), \
        "No DeprecationWarning raised"
    assert "deprecated" in str(w[0].message).lower(), \
        f"Warning message doesn't mention 'deprecated': {w[0].message}"

print(f"  Warning message: {w[0].message}")
print(f"  quasi_likelihood_loss result: {loss_old.item():.6f}")
print("  [OK] Deprecation warning shown")

print("\n[Test 3] Both functions produce identical results...")
assert torch.allclose(loss_new, loss_old, atol=1e-6), \
    f"Results differ: new={loss_new.item():.8f}, old={loss_old.item():.8f}"
print(f"  Difference: {abs(loss_new.item() - loss_old.item()):.10f}")
print("  [OK] Functions produce identical results")

print("\n" + "="*70)
print("  ALL TESTS PASSED")
print("="*70)
print("\n[SUCCESS] Function rename is working correctly:")
print("  - gnnhar_ratio_loss is the new, preferred function")
print("  - quasi_likelihood_loss still works but warns about deprecation")
print("  - Both functions produce identical results")
print("  - Migration path is clear and safe")
