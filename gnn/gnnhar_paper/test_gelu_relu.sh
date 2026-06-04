#!/bin/bash
# Test GELU vs ReLU activation functions
# Small-scale test: 5 seeds, 100 epochs
# Expected: GELU should show +2-5% R² improvement

echo "=========================================="
echo "  GELU vs ReLU Activation Test (v1.1_GELU)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: GNNHAR1L"
echo "  Seeds: 5"
echo "  Epochs: 100"
echo "  Horizon: h=5"
echo ""

# Test 1: ReLU (baseline)
echo "=========================================="
echo "  Test 1: ReLU Activation (Baseline)"
echo "=========================================="
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --n_seeds 5 \
    --epochs 100 \
    --n_hid 16 \
    --activation relu \
    --horizon 5

echo ""
echo "=========================================="
echo "  Test 2: GELU Activation (Experimental)"
echo "=========================================="
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --n_seeds 5 \
    --epochs 100 \
    --n_hid 16 \
    --activation gelu \
    --horizon 5

echo ""
echo "=========================================="
echo "  Test Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/gnnhar_paper/multi_stock/"
echo "Compare files: GNNHAR1L_relu_h5_*.json vs GNNHAR1L_gelu_h5_*.json"
echo ""
echo "Expected: GELU should show +2-5% R² improvement over ReLU"
echo "If improvement > 5%, proceed to full ensemble training (20 seeds, 400 epochs)"
