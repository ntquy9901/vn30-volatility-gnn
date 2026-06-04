"""Test VCI API - refined historical depth test with proper timestamp parsing."""
import sys
import requests
import datetime

sys.stdout.reconfigure(line_buffering=True)

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://vietcap.com.vn",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

url = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"


def parse_ts(v):
    """Parse timestamp: int unix or str datetime."""
    if isinstance(v, (int, float)):
        return datetime.datetime.fromtimestamp(v).strftime("%Y-%m-%d %H:%M")
    if isinstance(v, str):
        return v[:16]
    return str(v)


def test_period(label, start_str, end_str, interval="ONE_MINUTE"):
    start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d") + datetime.timedelta(days=1)
    bdays = sum(1 for i in range((end_dt - start_dt).days)
                if (start_dt + datetime.timedelta(i)).weekday() < 5)
    bars_per_day = 255 if interval == "ONE_MINUTE" else 1
    count_back = bdays * bars_per_day + 10
    to_stamp = int(end_dt.timestamp())

    payload = {"timeFrame": interval, "symbols": ["VCB"],
               "to": to_stamp, "countBack": count_back}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code != 200:
            print("  HTTP%d [    0] %s" % (r.status_code, label))
            return 0
        d = r.json()
        item = d[0] if isinstance(d, list) and len(d) > 0 else d
        if not isinstance(item, dict):
            print("  EMPTY [    0] %s" % label)
            return 0
        t_arr = item.get("t", [])
        n = len(t_arr) if isinstance(t_arr, list) else 0
        if n == 0:
            print("  EMPTY [    0] %s" % label)
            return 0
        status = "OK   "
        oldest = parse_ts(t_arr[-1])
        newest = parse_ts(t_arr[0])
        print("  %s [%5d] %s  |  %s -> %s" % (status, n, label, oldest, newest))
        return n
    except Exception as e:
        print("  ERR   [    0] %s: %s" % (label, str(e)[:80]))
        return 0


print("=" * 70)
print("VCI 1-min historical depth (VCB) -- binary search for cutoff")
print("=" * 70)
test_period("2026-05 (recent)", "2026-05-15", "2026-05-20")
test_period("2026-01", "2026-01-06", "2026-01-10")
test_period("2025-06", "2025-06-02", "2025-06-06")
test_period("2025-01", "2025-01-06", "2025-01-10")
test_period("2024-07", "2024-07-01", "2024-07-05")
test_period("2024-04", "2024-04-01", "2024-04-05")
test_period("2024-01", "2024-01-02", "2024-01-06")
test_period("2023-10", "2023-10-02", "2023-10-06")
test_period("2023-07", "2023-07-03", "2023-07-07")
test_period("2023-04", "2023-04-03", "2023-04-07")
test_period("2023-01", "2023-01-02", "2023-01-06")
test_period("2022-07", "2022-07-04", "2022-07-08")
test_period("2022-01", "2022-01-03", "2022-01-07")

print()
print("VCI 1D (full history check):")
test_period("2009-01", "2009-01-05", "2009-01-09", interval="ONE_DAY")
test_period("2007-01", "2007-01-08", "2007-01-12", interval="ONE_DAY")
test_period("2005-01", "2005-01-03", "2005-01-07", interval="ONE_DAY")
