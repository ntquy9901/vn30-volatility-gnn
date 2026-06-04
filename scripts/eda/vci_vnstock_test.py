"""Test VCI intraday via vnstock library (which handles auth headers)."""
import sys
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# Suppress vnstock banner
import os
os.environ["VNSTOCK_SUPPRESS_BANNER"] = "1"

from vnstock import Quote

periods = [
    ("2026-05 recent", "2026-05-15", "2026-05-20"),
    ("2025-01", "2025-01-06", "2025-01-10"),
    ("2024-07", "2024-07-01", "2024-07-05"),
    ("2024-01", "2024-01-02", "2024-01-06"),
    ("2023-07", "2023-07-03", "2023-07-07"),
    ("2023-01", "2023-01-02", "2023-01-06"),
    ("2022-01", "2022-01-03", "2022-01-07"),
    ("2020-01", "2020-01-02", "2020-01-06"),
    ("2015-01", "2015-01-05", "2015-01-09"),
]

print("=" * 70)
print("VCI 5-min via vnstock Quote(source='VCI')")
print("=" * 70)

q = Quote(symbol="VCB", source="VCI", show_log=False)

for label, start, end in periods:
    try:
        df = q.history(start=start, end=end, interval="5m", show_log=False)
        n = len(df)
        status = "OK   " if n > 0 else "EMPTY"
        if n > 0:
            oldest = str(df["time"].iloc[-1])[:16]
            newest = str(df["time"].iloc[0])[:16]
            print("  %s [%5d bars] %s  |  %s -> %s" % (status, n, label, oldest, newest))
        else:
            print("  %s [    0 bars] %s" % (status, label))
    except Exception as e:
        print("  ERR   [    0 bars] %s: %s" % (label, str(e)[:80]))

print()
print("VCI 1D via vnstock:")
for label, start, end in [("2006-01", "2006-01-02", "2006-01-10"),
                            ("2000-01", "2000-01-02", "2000-01-10")]:
    try:
        df = q.history(start=start, end=end, interval="1D", show_log=False)
        n = len(df)
        status = "OK   " if n > 0 else "EMPTY"
        if n > 0:
            print("  %s [%5d bars] %s  |  %s -> %s" % (
                status, n, label, str(df["time"].iloc[0])[:10], str(df["time"].iloc[-1])[:10]))
        else:
            print("  %s [    0 bars] %s" % (status, label))
    except Exception as e:
        print("  ERR   [    0 bars] %s: %s" % (label, str(e)[:80]))
