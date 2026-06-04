#!/bin/bash
# Optuna Hyperparameter Optimization for GNNHAR1L
# Run this script from the gnn/gnnhar_paper directory

echo "=========================================="
echo "  Optuna Optimization: GNNHAR1L (GELU)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: GNNHAR1L"
echo "  Trials: 100"
echo "  Max epochs: 200 per trial"
echo "  Activation: GELU"
echo "  Horizon: h=5"
echo ""

# Check if Optuna is installed
if ! python -c "import optuna" 2>/dev/null; then
    echo "[Step 1] Installing Optuna..."
    pip install optuna optuna-dashboard
else
    echo "[Step 1] Optuna already installed"
fi

echo ""
echo "[Step 2] Running Optuna optimization..."
echo "This will take ~8 hours (100 trials)"
echo ""

# Run from current directory (where this script is located)
python optuna_gnnhar_optimization.py \
    --model GNNHAR1L \
    --activation gelu \
    --n_trials 100 \
    --epochs 200 \
    --horizon 5 \
    --device cpu

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Optimization failed. Check error message above."
    echo ""
    exit 1
fi

echo ""
echo "=========================================="
echo "  Optimization Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/gnnhar_paper/optuna/"
echo ""

# Find and display the most recent JSON file
LATEST_FILE=$(ls -t results/gnnhar_paper/optuna/GNNHAR1L_gelu_optuna_*.json 2>/dev/null | head -1)

if [ -n "$LATEST_FILE" ]; then
    echo "Latest results: $LATEST_FILE"
    echo ""
    echo "To view best hyperparameters:"
    echo "  cat results/gnnhar_paper/optuna/$LATEST_FILE"
    echo ""
fi

echo "Next steps:"
echo "1. Check best hyperparameters in the JSON file"
echo "2. Train final model with best hyperparameters:"
echo ""
echo "   python train_multi_stock.py --model GNNHAR1L --activation gelu --n_seeds 20 --epochs 400 --lr BEST_LR --weight_decay BEST_WD --n_hid BEST_HID --adj_threshold BEST_THRESH"
echo ""
