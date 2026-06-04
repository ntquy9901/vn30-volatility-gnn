@echo off
REM Optuna Hyperparameter Optimization for GNNHAR1L
REM Optimizes: lr, weight_decay, n_hid, adj_threshold, dropout
REM Expected: +5-10%% R² improvement
REM Duration: ~8 hours (100 trials)

echo ==========================================
echo   Optuna Optimization: GNNHAR1L (GELU)
echo ==========================================
echo.
echo Configuration:
echo   Model: GNNHAR1L
echo   Trials: 100
echo   Max epochs: 200 per trial
echo   Activation: GELU
echo   Horizon: h=5
echo.

REM Check if Optuna is installed
python -c "import optuna" 2>nul
if errorlevel 1 (
    echo [ERROR] Optuna not installed. Installing...
    pip install optuna
)

echo [Step 1] Installing Optuna (if not already installed)...
pip install optuna optuna-dashboard

echo.
echo [Step 2] Running Optuna optimization...
echo This will take ~8 hours (100 trials)
echo.

REM Change to project root directory (2 levels up from current script location)
cd %~dp0\..\..\..

python gnn\gnnhar_paper\optuna_gnnhar_optimization.py ^
    --model GNNHAR1L ^
    --activation gelu ^
    --n_trials 100 ^
    --epochs 200 ^
    --horizon 5 ^
    --device cpu

echo.
echo ==========================================
echo   Optimization Complete!
echo ==========================================
echo.
echo Results saved to: results\gnnhar_paper\optuna\
echo.
echo Next steps:
echo 1. Check best hyperparameters in JSON file
echo 2. Train final model with best hyperparameters:
echo    cd D:\bmad-projects\luanvan_exp\moirai
echo    python gnn\gnnhar_paper\train_multi_stock.py ^
echo        --model GNNHAR1L ^
echo        --activation gelu ^
echo        --n_seeds 20 ^
echo        --epochs 400 ^
echo        --lr BEST_LR ^
echo        --weight_decay BEST_WD ^
echo        --n_hid BEST_HID ^
echo        --adj_threshold BEST_THRESH
echo.
pause
