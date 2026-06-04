"""Test VNDirect dchart API for intraday historical depth."""
import sys
import requests
import datetime

sys.stdout.reconfigure(line_buffering=True)

h = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://vndirect.com.vn/",
    "Origin": "https://vndirect.com.vn",
}

base = "https://dchart-api.vndirect.com.vn/dchart/history"


def test_range(label, year, month, day_s, day_e, res="5"):
    from_ts = int(datetime.datetime(year, month, day_s).timestamp())
    to_ts = int(datetime.datetime(year, month, day_e, 23, 59).timestamp())
    url = "%s?resolution=%s&symbol=VCB&from=%d&to=%d" % (base, res, from_ts, to_ts)
    try:
        r = requests.get(url, headers=h, timeout=10)
        if r.status_code == 200:
            d = r.json()
            t_arr = d.get("t", []) if isinstance(d, dict) else []
            n = len(t_arr) if isinstance(t_arr, list) else 0
            s_val = d.get("s", "?")
            st = "OK   " if n > 0 else "EMPTY"
            print("  %s [%5d bars] res=%s %s (s=%s)" % (st, n, res, label, s_val))
            if n > 0:
                def parse_t(v):
                    if isinstance(v, (int, float)):
                        return datetime.datetime.fromtimestamp(v).strftime("%Y-%m-%d %H:%M")
                    return str(v)[:16]
                print("       first=%s  last=%s" % (parse_t(t_arr[0]), parse_t(t_arr[-1])))
        else:
            print("  HTTP%d [    0 bars] res=%s %s" % (r.status_code, res, label))
    except Exception as e:
        print("  ERR   [    0 bars] res=%s %s: %s" % (res, label, str(e)[:80]))


print("VNDirect dchart 5-min depth (VCB):")
test_range("2026-05-15..22", 2026, 5, 15, 22)
test_range("2026-01-06..10", 2026, 1, 6, 10)
test_range("2025-11-03..07", 2025, 11, 3, 7)
test_range("2025-06-02..06", 2025, 6, 2, 6)
test_range("2025-01-06..10", 2025, 1, 6, 10)
test_range("2024-07-01..05", 2024, 7, 1, 5)
test_range("2024-01-02..06", 2024, 1, 2, 6)
test_range("2023-07-03..07", 2023, 7, 3, 7)
test_range("2023-01-02..06", 2023, 1, 2, 6)
test_range("2022-01-03..07", 2022, 1, 3, 7)
test_range("2020-01-02..06", 2020, 1, 2, 6)
test_range("2018-01-02..06", 2018, 1, 2, 6)
test_range("2015-01-05..09", 2015, 1, 5, 9)

print()
print("VNDirect dchart Daily depth (VCB):")
test_range("2010-01-04..08", 2010, 1, 4, 8, res="D")
test_range("2006-01-02..06", 2006, 1, 2, 6, res="D")
test_range("2000-01-03..07", 2000, 1, 3, 7, res="D")
