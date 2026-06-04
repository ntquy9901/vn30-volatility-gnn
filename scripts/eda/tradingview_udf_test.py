"""Test TradingView UDF-format chart APIs on Vietnamese financial sites."""
import sys
import requests
import datetime

sys.stdout.reconfigure(line_buffering=True)

h_base = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}

def mk_ts(y, m=1, d=2):
    return int(datetime.datetime(y, m, d).timestamp())

# Time ranges to probe
ranges = [
    ("2026-05", mk_ts(2026, 5, 15), mk_ts(2026, 5, 22)),
    ("2024-01", mk_ts(2024, 1, 2), mk_ts(2024, 1, 9)),
    ("2020-01", mk_ts(2020, 1, 2), mk_ts(2020, 1, 9)),
    ("2015-01", mk_ts(2015, 1, 5), mk_ts(2015, 1, 12)),
]


def test_tv_endpoint(site, base_url, referer, resolutions=None, symbol="VCB"):
    """Test a TradingView UDF-style endpoint for multiple time ranges."""
    if resolutions is None:
        resolutions = ["5", "D"]
    print("\n--- %s ---" % site)
    h = dict(h_base)
    h["Referer"] = referer

    for res in resolutions[:1]:  # Test 5-min first
        for label, from_ts, to_ts in ranges[:2]:  # recent + 2024
            url = "%s?symbol=%s&resolution=%s&from=%d&to=%d" % (
                base_url, symbol, res, from_ts, to_ts)
            try:
                r = requests.get(url, headers=h, timeout=10)
                if r.status_code == 200:
                    ct = r.headers.get("Content-Type", "")
                    if "html" in ct.lower():
                        print("  [HTML] %s res=%s -- returns HTML (needs auth?)" % (label, res))
                        break
                    try:
                        d = r.json()
                        if isinstance(d, dict):
                            t_arr = d.get("t", [])
                            n = len(t_arr) if isinstance(t_arr, list) else 0
                            s_val = d.get("s", "?")
                            if n > 0:
                                print("  [OK  ] %s res=%s n=%d s=%s" % (label, res, n, s_val))
                            else:
                                print("  [EMPT] %s res=%s s=%s keys=%s" % (
                                    label, res, s_val, list(d.keys())[:6]))
                        else:
                            print("  [OK  ] %s res=%s type=%s" % (label, res, type(d).__name__))
                    except Exception:
                        print("  [RAW ] %s res=%s body=%s" % (label, res, r.text[:80]))
                else:
                    print("  [%d  ] %s res=%s" % (r.status_code, label, res))
            except Exception as e:
                print("  [ERR ] %s res=%s: %s" % (label, res, str(e)[:60]))


# SSI iboard TradingView datafeed
test_tv_endpoint(
    "SSI iboard (TradingView UDF)",
    "https://iboard.ssi.com.vn/dchart/api/history",
    "https://iboard.ssi.com.vn/",
)

# BSC (BIDV Securities) chart
test_tv_endpoint(
    "BSC chart",
    "https://chart.bsc.com.vn/api/history",
    "https://bsc.com.vn/",
)

# VCSC (VietCapital Securities)
test_tv_endpoint(
    "VCSC chart",
    "https://chart.vcsc.com.vn/api/history",
    "https://vcsc.com.vn/",
)

# Simplize.vn
test_tv_endpoint(
    "Simplize.vn chart",
    "https://chart.simplize.vn/api/history",
    "https://simplize.vn/",
)

# StockBiz
test_tv_endpoint(
    "StockBiz chart",
    "https://api.stockbiz.vn/chart/history",
    "https://stockbiz.vn/",
)

# WiChart
test_tv_endpoint(
    "WiChart",
    "https://api.wichart.vn/chart/history",
    "https://wichart.vn/",
)

# Fialda (formerly StockVN)
test_tv_endpoint(
    "Fialda chart",
    "https://api.fialda.com/chart/history",
    "https://fialda.com/",
)

# TradingEconomy / StockQ
test_tv_endpoint(
    "StockQ chart (TradingView)",
    "https://chart.stockq.org/api/history",
    "https://stockq.org/",
)
