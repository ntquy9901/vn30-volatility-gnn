"""Test KBS API historical depth for 5-min intraday data."""
import sys
import requests

sys.stdout.reconfigure(line_buffering=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}

base = "https://kbbuddywts.kbsec.com.vn/iis-server/investment"

periods = [
    ("3m ago  2026-02", "03-02-2026", "07-02-2026"),
    ("4m ago  2026-01", "05-01-2026", "09-01-2026"),
    ("5m ago  2025-12", "01-12-2025", "05-12-2025"),
    ("6m ago  2025-11", "03-11-2025", "07-11-2025"),
    ("8m ago  2025-09", "01-09-2025", "05-09-2025"),
    ("12m ago 2025-05", "05-05-2025", "09-05-2025"),
    ("18m ago 2024-11", "04-11-2024", "08-11-2024"),
    ("24m ago 2024-05", "06-05-2024", "10-05-2024"),
]

print("KBS 5-min historical depth (VCB):")
for label, sdate, edate in periods:
    url = "%s/stocks/VCB/data_5P?sdate=%s&edate=%s" % (base, sdate, edate)
    try:
        r = requests.get(url, headers=headers, timeout=10)
        d = r.json() if r.status_code == 200 else {}
        bars = d.get("data_5P", [])
        n = len(bars) if isinstance(bars, list) else 0
        status = "OK   " if n > 0 else "EMPTY"
        print("  %s [%4d bars] %s" % (status, n, label))
        if n > 0 and isinstance(bars, list):
            oldest = bars[-1].get("t", "?")
            newest = bars[0].get("t", "?")
            print("       oldest=%s  newest=%s" % (oldest, newest))
    except Exception as e:
        print("  ERR  [   0 bars] %s: %s" % (label, e))

# Also test VCI source
print("\nVCI 5-min historical depth (VCB):")
vci_base = "https://api.vietstock.vn"

# Try vietstock
periods2 = [
    ("recent 2026-05", "2026-05-15", "2026-05-20"),
    ("2025-01", "2025-01-02", "2025-01-07"),
    ("2024-01", "2024-01-02", "2024-01-07"),
    ("2020-01", "2020-01-02", "2020-01-07"),
]

# VietStock chart API (used by vietstock.vn)
vs_headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://vietstock.vn/",
}
for label, s, e in periods2:
    # Timestamps
    import datetime
    ts = int(datetime.datetime.strptime(s, "%Y-%m-%d").timestamp())
    te = int(datetime.datetime.strptime(e, "%Y-%m-%d").timestamp())
    url = "https://iboard.ssi.com.vn/dchart/api/history?resolution=5&symbol=VCB&from=%d&to=%d" % (ts, te)
    try:
        r = requests.get(url, headers=vs_headers, timeout=10)
        d = r.json() if r.status_code == 200 else {}
        t_arr = d.get("t", [])
        n = len(t_arr) if isinstance(t_arr, list) else 0
        status = "OK   " if n > 0 else "EMPTY"
        print("  %s [%4d bars] SSI iboard %s" % (status, n, label))
        if n > 0:
            import datetime as dt2
            oldest = dt2.datetime.fromtimestamp(t_arr[-1]).strftime("%Y-%m-%d %H:%M")
            newest = dt2.datetime.fromtimestamp(t_arr[0]).strftime("%Y-%m-%d %H:%M")
            print("       oldest=%s  newest=%s" % (oldest, newest))
    except Exception as e2:
        print("  ERR  [   0 bars] SSI iboard %s: %s" % (label, e2))
