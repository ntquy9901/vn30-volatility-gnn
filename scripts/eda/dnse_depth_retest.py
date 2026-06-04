"""DNSE historical depth retest with correct date format."""
import sys
import requests
import datetime

sys.stdout.reconfigure(line_buffering=True)

h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
base = "https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"

periods = [
    ("2026-05 (recent)", "2026-05-15", "2026-05-22"),
    ("2026-01", "2026-01-06", "2026-01-10"),
    ("2025-11", "2025-11-03", "2025-11-10"),
    ("2025-09", "2025-09-01", "2025-09-08"),
    ("2025-06", "2025-06-02", "2025-06-09"),
    ("2025-03", "2025-03-03", "2025-03-10"),
    ("2025-01", "2025-01-06", "2025-01-13"),
    ("2024-10", "2024-10-01", "2024-10-08"),
    ("2024-07", "2024-07-01", "2024-07-08"),
    ("2024-04", "2024-04-01", "2024-04-08"),
    ("2024-01", "2024-01-02", "2024-01-09"),
    ("2023-10", "2023-10-02", "2023-10-09"),
    ("2023-07", "2023-07-03", "2023-07-10"),
    ("2023-01", "2023-01-02", "2023-01-09"),
    ("2022-01", "2022-01-03", "2022-01-10"),
    ("2021-01", "2021-01-04", "2021-01-11"),
    ("2020-01", "2020-01-02", "2020-01-09"),
    ("2018-01", "2018-01-02", "2018-01-09"),
    ("2015-01", "2015-01-05", "2015-01-12"),
    ("2010-01", "2010-01-04", "2010-01-11"),
]

resolutions = ["5", "1", "15", "D"]

print("DNSE 5-min depth test (VCB):")
for label, s, e in periods:
    url = "%s?from=%s&to=%s&symbol=VCB&resolution=5" % (base, s, e)
    try:
        r = requests.get(url, headers=h, timeout=10)
        if r.status_code == 200:
            d = r.json()
            t_arr = d.get("t", []) if isinstance(d, dict) else []
            n = len(t_arr) if isinstance(t_arr, list) else 0
            st = "OK   " if n > 0 else "EMPTY"
            print("  %s [%5d] %s" % (st, n, label))
            if n > 0:
                def parse_t(v):
                    if isinstance(v, (int, float)):
                        return datetime.datetime.fromtimestamp(v).strftime("%Y-%m-%d %H:%M")
                    return str(v)[:16]
                print("       first=%s  last=%s" % (parse_t(t_arr[0]), parse_t(t_arr[-1])))
        else:
            print("  HTTP%d [    0] %s" % (r.status_code, label))
    except Exception as e2:
        print("  ERR   [    0] %s: %s" % (label, str(e2)[:80]))
