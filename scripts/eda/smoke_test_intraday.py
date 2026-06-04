"""
Smoke test for Vietnamese stock intraday data sources.
Run: python scripts/eda/smoke_test_intraday.py
"""
import sys
import requests
import json
import time

sys.stdout.reconfigure(line_buffering=True)

headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# Unix timestamps
TS_2024_JAN_02 = 1704067200  # 2024-01-02 00:00:00 UTC
TS_2024_JAN_05 = 1704326400  # 2024-01-05 00:00:00 UTC
TS_2020_JAN_02 = 1577836800
TS_2020_JAN_10 = 1578614400
TS_2015_JAN_02 = 1420156800
TS_2015_JAN_10 = 1420934400


def test_url(label, url, timeout=10):
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        body = r.text[:300]
        print(f"  [{r.status_code}] {label}: {url[:80]}")
        if r.status_code == 200:
            try:
                d = r.json()
                if isinstance(d, list):
                    print(f"    -> list len={len(d)}", end="")
                    if len(d) > 0 and isinstance(d[0], dict):
                        print(f", keys={list(d[0].keys())[:6]}")
                    else:
                        print()
                elif isinstance(d, dict):
                    print(f"    -> dict keys={list(d.keys())[:8]}")
                    for k in ["data", "d", "bars", "candles", "ohlc"]:
                        if k in d and d[k]:
                            bars = d[k]
                            print(f"    -> d['{k}']: len={len(bars) if hasattr(bars, '__len__') else 'N/A'}")
                            if isinstance(bars, list) and len(bars) > 0:
                                print(f"       first={str(bars[0])[:100]}")
                            break
            except Exception as e:
                print(f"    -> json parse error: {e}, body={body[:100]}")
        else:
            print(f"    -> body={body[:150]}")
    except Exception as e:
        print(f"  [ERR] {label}: {e}")


print("=" * 70)
print("  Smoke Test: Vietnamese Stock Intraday Data Sources")
print("=" * 70)

# ---- 1. SSI FastConnect public endpoints ----
print("\n--- SSI (public no-auth) ---")
test_url("SSI chart 5min 2024", f"https://iboard-query.ssi.com.vn/v2/stock/his/paging?symbol=VCB&fromDate=2024-01-02&toDate=2024-01-05&size=100")
test_url("SSI pricehist", f"https://iboard.ssi.com.vn/statistics/stock-price/VCB?type=1&fromDate=20240102&toDate=20240105")
test_url("SSI candlestick", f"https://iboard-query.ssi.com.vn/v1/market/chart?symbol=VCB&resolution=5&from={TS_2024_JAN_02}&to={TS_2024_JAN_05}")

# ---- 2. Fireant ----
print("\n--- Fireant (no auth) ---")
test_url("Fireant hist quotes", "https://restv2.fireant.vn/symbols/VCB/historical-quotes?startDate=2024-01-02&endDate=2024-01-05&offset=0&limit=100")
test_url("Fireant chart 5min", "https://restv2.fireant.vn/symbols/VCB/chart-data?startDate=2024-01-02T09:00:00&endDate=2024-01-05T15:00:00&resolution=5")
test_url("Fireant fundamental", "https://restv2.fireant.vn/symbols/VCB/fundamental")

# ---- 3. VNDirect ----
print("\n--- VNDirect ---")
test_url("VNDirect daily", "https://finfo-api.vndirect.com.vn/v4/stock_prices?code=VCB&sort=date&size=5&page=1&fromDate=2024-01-02&toDate=2024-01-05")
test_url("VNDirect intraday", f"https://finfo-api.vndirect.com.vn/v4/stock_prices/intraday?code=VCB&from={TS_2024_JAN_02}&to={TS_2024_JAN_05}&resolution=5")

# ---- 4. iTick ----
print("\n--- iTick ---")
test_url("iTick candles 5min", f"https://api.itick.vn/stock/history?sym=VCB&resolution=5&from={TS_2024_JAN_02}&to={TS_2024_JAN_05}")
test_url("iTick candles 5min v2", f"https://api.itick.vn/candles?symbol=VCB&resolution=5&from={TS_2024_JAN_02}&to={TS_2024_JAN_05}")

# ---- 5. WIFEED / wifeed.vn ----
print("\n--- WiFeed ---")
test_url("WiFeed daily", "https://api.wifeed.vn/api/stock/getStockHistory?stockCode=VCB&startDate=2024-01-02&endDate=2024-01-05")

# ---- 6. AlgoTrade ----
print("\n--- AlgoTrade ---")
test_url("AlgoTrade candles", f"https://api.algotrade.vn/data/stock/history?symbol=VCB&resolution=5&from={TS_2024_JAN_02}&to={TS_2024_JAN_05}")

# ---- 7. DNSE extended test (historical depth) ----
print("\n--- DNSE historical depth test ---")
dnse_base = "https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
for year in [2026, 2024, 2022, 2020, 2018, 2015]:
    from_ts = f"{year}-01-02T09:00:00"
    to_ts = f"{year}-01-06T15:00:00"
    url = f"{dnse_base}?from={from_ts}&to={to_ts}&symbol=VCB&resolution=5"
    try:
        r = requests.get(url, headers=headers, timeout=8)
        d = r.json() if r.status_code == 200 else {}
        bars = d.get("t", []) if isinstance(d, dict) else []
        print(f"  DNSE {year} 5min: status={r.status_code}, bars={len(bars) if bars else 0}")
        if bars and len(bars) > 0:
            print(f"    first timestamp: {bars[0]}")
    except Exception as e:
        print(f"  DNSE {year}: ERR {e}")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
