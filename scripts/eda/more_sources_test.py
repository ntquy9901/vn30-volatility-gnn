"""Test more Vietnamese stock intraday data sources."""
import sys
import requests
import datetime

sys.stdout.reconfigure(line_buffering=True)

h = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}

ts_2024 = int(datetime.datetime(2024, 1, 2, 9, 0).timestamp())
te_2024 = int(datetime.datetime(2024, 1, 6, 15, 0).timestamp())
ts_2020 = int(datetime.datetime(2020, 1, 2, 9, 0).timestamp())
te_2020 = int(datetime.datetime(2020, 1, 6, 15, 0).timestamp())
ts_2015 = int(datetime.datetime(2015, 1, 5, 9, 0).timestamp())
te_2015 = int(datetime.datetime(2015, 1, 9, 15, 0).timestamp())


def try_get(label, url, hdrs=None, timeout=10):
    hh = dict(h)
    if hdrs:
        hh.update(hdrs)
    try:
        r = requests.get(url, headers=hh, timeout=timeout)
        body = r.text[:200]
        print("[%d] %s" % (r.status_code, label))
        if r.status_code == 200 and body.strip().startswith("{"):
            d = r.json()
            if isinstance(d, dict):
                print("    keys=%s" % list(d.keys())[:8])
                # look for arrays
                for k, v in d.items():
                    if isinstance(v, list) and len(v) > 0:
                        print("    d['%s']: %d items, first=%s" % (k, len(v), str(v[0])[:80]))
                        break
            elif isinstance(d, list):
                print("    list n=%d" % len(d))
        elif r.status_code == 200 and body.strip().startswith("["):
            import json
            arr = r.json()
            print("    list n=%d" % len(arr))
            if arr and isinstance(arr[0], dict):
                print("    first keys=%s" % list(arr[0].keys())[:8])
        else:
            print("    body=%s" % body[:100])
    except Exception as e:
        print("[ERR] %s: %s" % (label, str(e)[:100]))


print("=" * 70)
print("DNSE connector via vnstock (direct test)")
print("=" * 70)
# The DNSE connector in vnstock
dnse_base = "https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
for year, from_s, to_s in [
    (2026, "2026-05-15T09:00:00", "2026-05-20T15:00:00"),
    (2025, "2025-01-06T09:00:00", "2025-01-10T15:00:00"),
    (2024, "2024-01-02T09:00:00", "2024-01-06T15:00:00"),
    (2023, "2023-01-02T09:00:00", "2023-01-06T15:00:00"),
    (2022, "2022-01-03T09:00:00", "2022-01-07T15:00:00"),
    (2020, "2020-01-02T09:00:00", "2020-01-06T15:00:00"),
    (2018, "2018-01-02T09:00:00", "2018-01-06T15:00:00"),
    (2015, "2015-01-05T09:00:00", "2015-01-09T15:00:00"),
]:
    url = "%s?from=%s&to=%s&symbol=VCB&resolution=5" % (dnse_base, from_s, to_s)
    try:
        r = requests.get(url, headers=h, timeout=10)
        if r.status_code == 200:
            d = r.json()
            t_arr = d.get("t", []) if isinstance(d, dict) else []
            n = len(t_arr) if isinstance(t_arr, list) else 0
            st = "OK   " if n > 0 else "EMPTY"
            print("  %s [%5d bars] DNSE 5m year=%d" % (st, n, year))
            if n > 0:
                import datetime as dt2
                ts0 = t_arr[0]
                ts1 = t_arr[-1]
                if isinstance(ts0, (int, float)):
                    t0s = dt2.datetime.fromtimestamp(ts0).strftime("%Y-%m-%d %H:%M")
                    t1s = dt2.datetime.fromtimestamp(ts1).strftime("%Y-%m-%d %H:%M")
                else:
                    t0s, t1s = str(ts0)[:16], str(ts1)[:16]
                print("       first=%s  last=%s" % (t0s, t1s))
        else:
            print("  HTTP%d [    0 bars] DNSE 5m year=%d" % (r.status_code, year))
    except Exception as e:
        print("  ERR   [    0 bars] DNSE 5m year=%d: %s" % (year, str(e)[:80]))

print()
print("=" * 70)
print("WiChart / BIDV Securities / HCM Securities chart APIs")
print("=" * 70)
try_get("HCMS chart 5m", "https://api2.hschart.vn/chart/history?symbol=VCB&resolution=5&from=%d&to=%d" % (ts_2024, te_2024))
try_get("VndRect chart", "https://trading.vndirect.com.vn/price-service/api/priceboard/intraday?code=VCB&from=2024-01-02&to=2024-01-06&resolution=5")
try_get("MBS chart", "https://api.mbs.com.vn/stock-service/chart/history?symbol=VCB&resolution=5&from=%d&to=%d" % (ts_2024, te_2024))

print()
print("=" * 70)
print("SSI FastConnect public data endpoint")
print("=" * 70)
try_get("SSI FC daily", "https://fc-data.ssi.com.vn/api/v2/Trading/PriceHistory?Symbol=VCB&Resolution=D&From=%d&To=%d" % (ts_2024, te_2024),
        hdrs={"Referer": "https://iboard.ssi.com.vn/"})
try_get("SSI FC 5min", "https://fc-data.ssi.com.vn/api/v2/Trading/PriceHistory?Symbol=VCB&Resolution=5&From=%d&To=%d" % (ts_2024, te_2024),
        hdrs={"Referer": "https://iboard.ssi.com.vn/"})

print()
print("=" * 70)
print("StockBiz / VNDirect data store")
print("=" * 70)
try_get("VNDirect price store", "https://finfo-api.vndirect.com.vn/v4/stock_prices?code=VCB&sort=-date&size=10&page=1&fromDate=2024-01-02&toDate=2024-01-06")
try_get("VNDirect intraday v4", "https://finfo-api.vndirect.com.vn/v4/intraday/price?code=VCB&date=2024-01-02&resolution=5")
