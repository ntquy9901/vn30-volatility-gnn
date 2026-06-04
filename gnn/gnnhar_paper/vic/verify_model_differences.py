"""
Verify that model fixes were applied correctly
Check if models are actually different after ReLU removal
"""
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn.gnnhar_paper.gnnhar_models import MODEL_REGISTRY

print("\n" + "="*70)
print("  VERIFYING MODEL FIXES")
print("="*70 + "\n")

# Create test input
batch_size = 2
n_stocks = 3
node_feat = torch.randn(batch_size, n_stocks, 3)
adj = torch.randn(n_stocks, n_stocks)

print("[Test Input]")
print(f"  node_feat shape: {node_feat.shape}")
print(f"  adj shape: {adj.shape}")
print(f"  node_feat mean: {node_feat.mean():.4f}")
print(f"  node_feat range: [{node_feat.min():.4f}, {node_feat.max():.4f}]")

print(f"\n{'='*70}")
print(f"  TESTING EACH MODEL")
print(f"{'='*70}\n")

for model_name in ['HAR', 'GHAR', 'GNNHAR1L', 'GNNHAR2L', 'GNNHAR3L']:
    print(f"[{model_name}]")
    if model_name == 'HAR':
        model = MODEL_REGISTRY[model_name]()
    else:
        model = MODEL_REGISTRY[model_name](n_hid=16)

    # Check model structure
    print(f"  Parameters: {model.count_params()}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(node_feat, adj)

    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean():.6f}")
    print(f"  Output std: {output.std():.6f}")
    print(f"  Output range: [{output.min():.6f}, {output.max():.6f}]")

    # Check for negative outputs
    n_negative = (output < 0).sum().item()
    total = output.numel()
    print(f"  Negative outputs: {n_negative}/{total} ({n_negative/total*100:.1f}%)")

    # Check for zero outputs
    n_zero = (output == 0).sum().item()
    print(f"  Zero outputs: {n_zero}/{total} ({n_zero/total*100:.1f}%)")

    print()

print(f"{'='*70}")
print(f"  DIAGNOSIS")
print(f"{'='*70}\n")

print("EXPECTED BEHAVIOR (after fix):")
print("  - All models should allow negative outputs (no ReLU on final output)")
print("  - Zero outputs should be 0% (no dying ReLU)")
print("  - Different models should produce different outputs")

print("\nIF YOU SEE:")
print("  - 100% zero outputs: ReLU still active (fix not applied)")
print("  - All models same output: Caching or import issue")
print("  - Negative outputs present: GOOD! Fix is working")

print(f"\n{'='*70}\n")
