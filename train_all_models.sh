#!/bin/bash
# Train all 3 GNNHAR models (HAR, GHAR, GNNHAR1L) with full ensemble
# Total estimated time: ~9 hours

echo "=========================================="
echo "  Training All GNNHAR Models"
echo "=========================================="
echo ""
echo "Configuration: 20 seeds, 400 epochs, batch_size 512"
echo "Estimated time per model: 2-3.3 hours"
echo "Total estimated time: ~9 hours"
echo ""
echo "Starting training..."
echo ""

# Model 1: HAR baseline
echo "=========================================="
echo "  Model 1/3: HAR (Baseline)"
echo "=========================================="
python gnn/gnnhar_paper/train_multi_stock.py \
    --model HAR \
    --n_seeds 20 \
    --epochs 400 \
    --batch_size 512

echo ""
echo "HAR training completed!"
echo ""

# Model 2: GHAR (linear spillover)
echo "=========================================="
echo "  Model 2/3: GHAR (Linear Spillover)"
echo "=========================================="
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GHAR \
    --n_seeds 20 \
    --epochs 400 \
    --batch_size 512

echo ""
echo "GHAR training completed!"
echo ""

# Model 3: GNNHAR1L (nonlinear spillover)
echo "=========================================="
echo "  Model 3/3: GNNHAR1L (Nonlinear Spillover)"
echo "=========================================="
python gnn/gnnhar_paper/train_multi_stock.py \
    --model GNNHAR1L \
    --n_seeds 20 \
    --epochs 400 \
    --batch_size 512

echo ""
echo "GNNHAR1L training completed!"
echo ""
echo "=========================================="
echo "  All Models Training Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/gnnhar_paper/multi_stock/"
echo ""
echo "To analyze results, run:"
echo "  python gnn/gnnhar_paper/analyze_final_results.py"
