"""
Load CSV tài chính → GluonTS PandasDataset + train/test split.

Cách dùng:
    python src/data_loader.py --csv data/raw/stock.csv --target Close --date Date --freq B
"""
import argparse
from pathlib import Path

import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split


FREQ_ALIASES = {
    "B": "B",   # business day
    "D": "D",   # calendar day
    "W": "W-FRI",  # weekly ending Friday
    "M": "MS",  # month start
    "H": "h",   # hourly
}


def load_financial_dataset(
    csv_path: str | Path,
    target_col: str = "Close",
    date_col: str = "Date",
    freq: str = "B",
) -> PandasDataset:
    """
    Đọc CSV tài chính và trả về GluonTS PandasDataset.

    Args:
        csv_path: Đường dẫn file CSV.
        target_col: Tên cột giá trị cần dự báo.
        date_col: Tên cột ngày tháng.
        freq: Tần suất dữ liệu ('B', 'D', 'W', 'M', 'H').

    Returns:
        PandasDataset sẵn sàng cho GluonTS / Moirai.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df[[date_col, target_col]].dropna()
    df = df.set_index(date_col)
    df.index.freq = pd.tseries.frequencies.to_offset(FREQ_ALIASES.get(freq, freq))

    if df.index.freq is None:
        df = df.asfreq(FREQ_ALIASES.get(freq, freq), method="ffill")

    ds = PandasDataset(dict(df[[target_col]]))
    return ds


def make_train_test_split(
    ds: PandasDataset,
    prediction_length: int,
    n_windows: int = 5,
):
    """
    Chia dataset thành train và test theo rolling window.

    Args:
        ds: PandasDataset đầu vào.
        prediction_length: Số bước dự báo mỗi window.
        n_windows: Số rolling window để đánh giá.

    Returns:
        Tuple (train_dataset, test_data generator).
    """
    total_offset = -(prediction_length * n_windows)
    train_ds, test_template = split(ds, offset=total_offset)
    test_data = test_template.generate_instances(
        prediction_length=prediction_length,
        windows=n_windows,
        distance=prediction_length,
    )
    return train_ds, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra data loader")
    parser.add_argument("--csv", required=True, help="Đường dẫn file CSV")
    parser.add_argument("--target", default="Close", help="Tên cột target")
    parser.add_argument("--date", default="Date", help="Tên cột ngày")
    parser.add_argument("--freq", default="B", help="Tần suất (B/D/W/M)")
    parser.add_argument("--pdt", type=int, default=20, help="Prediction length")
    args = parser.parse_args()

    print(f"Đang tải: {args.csv}")
    ds = load_financial_dataset(args.csv, args.target, args.date, args.freq)

    df_check = pd.read_csv(args.csv)
    print(f"Tổng số dòng: {len(df_check)}")
    print(f"Cột target: {args.target} | Tần suất: {args.freq}")

    _, test_data = make_train_test_split(ds, prediction_length=args.pdt, n_windows=5)
    test_list = list(test_data.input)
    print(f"Số window test: {len(test_list)}")
    print(f"Context length window đầu: {len(test_list[0]['target'])}")
    print("Data loader OK.")
