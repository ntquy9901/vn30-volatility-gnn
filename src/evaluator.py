"""
Tính các metric đánh giá dự báo: MAE, MSE, MASE, CRPS.

Dùng GluonTS Evaluator để đảm bảo tính chuẩn xác theo chuẩn học thuật.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import SampleForecast


def compute_metrics(
    forecasts: list[SampleForecast],
    test_data,
    freq: str = "B",
    num_series: int | None = None,
) -> pd.DataFrame:
    """
    Tính MAE, MSE, MASE, CRPS từ forecasts và ground truth.

    Args:
        forecasts: List SampleForecast từ Moirai predictor.
        test_data: Test data iterator (từ generate_instances).
        freq: Tần suất dữ liệu.
        num_series: Số series để evaluate (None = tất cả).

    Returns:
        DataFrame với các cột: MAE, MSE, MASE, CRPS (mean trên tất cả windows).
    """
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

    agg_metrics, item_metrics = evaluator(
        ts_iterator=(entry["target"] for entry in test_data.label),
        fcst_iterator=iter(forecasts),
        num_series=num_series,
    )

    metrics = {
        "MAE": agg_metrics.get("mean_absolute_error", agg_metrics.get("MAE")),
        "MSE": agg_metrics.get("MSE"),
        "MASE": agg_metrics.get("MASE"),
        "CRPS": agg_metrics.get("mean_wQuantileLoss", agg_metrics.get("CRPS")),
        "RMSE": np.sqrt(agg_metrics.get("MSE", 0)),
    }

    return pd.DataFrame([metrics])


def compute_metrics_manual(
    forecasts: list[SampleForecast],
    actuals: list[np.ndarray],
    seasonal_period: int = 5,
) -> pd.DataFrame:
    """
    Tính metric thủ công khi GluonTS Evaluator không khả dụng.

    Args:
        forecasts: List SampleForecast.
        actuals: List array ground truth tương ứng.
        seasonal_period: Chu kỳ seasonal cho MASE (5 = tuần làm việc).

    Returns:
        DataFrame với MAE, MSE, RMSE, MASE, CRPS.
    """
    mae_list, mse_list, mase_list, crps_list = [], [], [], []

    for fcst, actual in zip(forecasts, actuals):
        median_pred = fcst.quantile(0.5)
        n = len(actual)

        mae = np.mean(np.abs(actual - median_pred))
        mse = np.mean((actual - median_pred) ** 2)
        mae_list.append(mae)
        mse_list.append(mse)

        # MASE: dùng naive seasonal forecast làm baseline
        naive_errors = np.abs(
            actual[seasonal_period:] - actual[:-seasonal_period]
        )
        scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        mase = mae / (scale + 1e-8)
        mase_list.append(mase)

        # CRPS xấp xỉ bằng pinball loss tại nhiều quantile
        quantiles = np.arange(0.05, 1.0, 0.05)
        crps_vals = []
        for q in quantiles:
            q_pred = fcst.quantile(q)
            pinball = np.mean(
                np.where(actual >= q_pred, q * (actual - q_pred), (1 - q) * (q_pred - actual))
            )
            crps_vals.append(pinball)
        crps_list.append(np.mean(crps_vals) * 2)

    metrics = {
        "MAE": np.mean(mae_list),
        "MSE": np.mean(mse_list),
        "RMSE": np.sqrt(np.mean(mse_list)),
        "MASE": np.mean(mase_list),
        "CRPS": np.mean(crps_list),
    }
    return pd.DataFrame([metrics])


def save_metrics(metrics_df: pd.DataFrame, output_path: str = "results/metrics.csv") -> None:
    """Lưu bảng metric ra file CSV."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Metrics đã lưu tại: {output_path}")
    print(metrics_df.to_string(index=False))
