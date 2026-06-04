"""Test VCI API historical depth for 1-min intraday data (5m = resampled from 1m)."""
import sys
import requests
import datetime

sys.stdout.reconfigure(line_buffering=True)

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://vietcap.com.vn",
    "Referer": "https://vietcap.com.vn/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

url = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"

def test_period(label, start_str, end_str, interval="ONE_MINUTE"):
    start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d") + datetime.timedelta(days=1)
    business_days = len([d for d in range((end_dt - start_dt).days)
                         if (start_dt + datetime.timedelta(d)).weekday() < 5])
    bars_per_day = 255 if interval == "ONE_MINUTE" else 1
    count_back = business_days * bars_per_day + 1
    to_stamp = int(end_dt.timestamp())

    payload = {
        "timeFrame": interval,
        "symbols": ["VCB"],
        "to": to_stamp,
        "countBack": count_back,
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            d = r.json()
            # VCI returns list or dict with 't' array
            if isinstance(d, list) and len(d) > 0:
                item = d[0]
                t_arr = item.get("t", [])
                n = len(t_arr) if isinstance(t_arr, list) else 0
                status = "OK   " if n > 0 else "EMPTY"
                print("  %s [%5d bars] %s" % (status, n, label))
                if n > 0:
                    oldest_ts = t_arr[-1]
                    newest_ts = t_arr[0]
                    oldest = datetime.datetime.fromtimestamp(oldest_ts).strftime("%Y-%m-%d %H:%M")
                    newest = datetime.datetime.fromtimestamp(newest_ts).strftime("%Y-%m-%d %H:%M")
                    print("       oldest=%s  newest=%s" % (oldest, newest))
            elif isinstance(d, dict):
                t_arr = d.get("t", [])
                n = len(t_arr) if isinstance(t_arr, list) else 0
                print("  dict [%5d bars] %s, keys=%s" % (n, label, list(d.keys())[:6]))
            else:
                print("  EMPTY [    0 bars] %s type=%s" % (label, type(d).__name__))
        else:
            print("  HTTP%d [    0 bars] %s: %s" % (r.status_code, label, r.text[:100]))
    except Exception as e:
        print("  ERR   [    0 bars] %s: %s" % (label, e))

print("VCI 1-min historical depth (VCB):")
test_period("recent 2026-05-15-20", "2026-05-15", "2026-05-20")
test_period("2026-04", "2026-04-01", "2026-04-05")
test_period("2026-01", "2026-01-06", "2026-01-10")
test_period("2025-11", "2025-11-03", "2025-11-07")
test_period("2025-09", "2025-09-01", "2025-09-05")
test_period("2025-06", "2025-06-02", "2025-06-06")
test_period("2025-01", "2025-01-06", "2025-01-10")
test_period("2024-01", "2024-01-02", "2024-01-06")
test_period("2023-01", "2023-01-02", "2023-01-06")
test_period("2020-01", "2020-01-02", "2020-01-06")
test_period("2015-01", "2015-01-05", "2015-01-09")

print("\nVCI 1D historical depth (VCB):")
test_period("2006-01", "2006-01-02", "2006-01-06", interval="ONE_DAY")
test_period("2010-01", "2010-01-04", "2010-01-08", interval="ONE_DAY")
test_period("2015-01", "2015-01-05", "2015-01-09", interval="ONE_DAY")
