"""Try various CafeF and Simplize endpoints for historical intraday data."""
import sys
import requests
import datetime

sys.stdout.reconfigure(line_buffering=True)

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi,en;q=0.9",
})

ts = {
    2026: (int(datetime.datetime(2026, 5, 15).timestamp()), int(datetime.datetime(2026, 5, 22).timestamp())),
    2024: (int(datetime.datetime(2024, 1, 2).timestamp()), int(datetime.datetime(2024, 1, 9).timestamp())),
    2020: (int(datetime.datetime(2020, 1, 2).timestamp()), int(datetime.datetime(2020, 1, 9).timestamp())),
    2015: (int(datetime.datetime(2015, 1, 5).timestamp()), int(datetime.datetime(2015, 1, 12).timestamp())),
}


def test(label, url, hdrs=None, timeout=10):
    h = {}
    if hdrs:
        h.update(hdrs)
    try:
        r = s.get(url, headers=h, timeout=timeout)
        print("[%d] %s" % (r.status_code, label))
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "")
            if "json" in ct.lower() or r.text.strip()[:1] in ("{", "["):
                try:
                    d = r.json()
                    if isinstance(d, list):
                        print("  list n=%d" % len(d))
                        if d and isinstance(d[0], dict):
                            print("  first keys=%s" % list(d[0].keys())[:8])
                    elif isinstance(d, dict):
                        print("  keys=%s" % list(d.keys())[:8])
                        for k in ["t", "data", "bars", "candles", "ohlc", "prices"]:
                            if k in d:
                                v = d[k]
                                n = len(v) if hasattr(v, "__len__") else "?"
                                print("  d['%s']: %s items" % (k, n))
                                if isinstance(v, list) and v:
                                    print("  first=%s" % str(v[0])[:100])
                                break
                except Exception as e:
                    print("  json err: %s" % e)
            else:
                print("  body (html)=%s..." % r.text[:80])
    except Exception as e:
        print("[ERR] %s: %s" % (label, str(e)[:80]))


print("=" * 70)
print("CafeF endpoints")
print("=" * 70)

# CafeF uses a proprietary chart system
cafef_h = {"Referer": "https://cafef.vn/"}

# Their stock page at cafef.vn/co-phieu-XXX.chn has a chart with historical data
# Try to find chart API endpoints
test("CafeF ajax intraday 2024-01-02", "https://cafef.vn/ajax/GetOHLCData.aspx?symbol=VCB&type=intraday&date=01/02/2024", cafef_h)
test("CafeF stock price history", "https://cafef.vn/ajax/RMS/AjaxStockHistory.aspx?symbol=VCB&startDate=02/01/2024&endDate=06/01/2024", cafef_h)
test("CafeF market chart v2", "https://cafef.vn/api/HistoricalData?symbol=VCB&resolution=5&from=%d&to=%d" % (ts[2024][0], ts[2024][1]), cafef_h)
test("CafeF priceboard chart", "https://cafef.vn/Ajax/GetDataChart2.aspx?symbol=VCB&period=5&date=20240102", cafef_h)

print()
print("=" * 70)
print("Simplize.vn endpoints (may require auth)")
print("=" * 70)
simplize_h = {"Referer": "https://simplize.vn/", "Origin": "https://simplize.vn"}
# Simplize uses GraphQL or REST
test("Simplize chart history", "https://api.simplize.vn/api/historical/chart?symbol=VCB&resolution=5&from=%d&to=%d" % (ts[2024][0], ts[2024][1]), simplize_h)
test("Simplize graphql", "https://api.simplize.vn/api/graphql", simplize_h)
test("Simplize chart ohlcv", "https://api.simplize.vn/api/stock/chart/ohlcv?symbol=VCB&interval=5m&from=2024-01-02&to=2024-01-06", simplize_h)

print()
print("=" * 70)
print("VNDirect iBoard (different from finfo-api)")
print("=" * 70)
test("VNDirect iboard chart", "https://iboard.vndirect.com.vn/chart/history?symbol=VCB&resolution=5&from=%d&to=%d" % (ts[2024][0], ts[2024][1]))
test("VNDirect market chart", "https://dchart-api.vndirect.com.vn/dchart/history?resolution=5&symbol=VCB&from=%d&to=%d" % (ts[2024][0], ts[2024][1]),
     hdrs={"Referer": "https://vndirect.com.vn/"})

print()
print("=" * 70)
print("TCBS new format (if API changed)")
print("=" * 70)
test("TCBS v3 chart", "https://apipubaws.tcbs.com.vn/stock-insight/v3/stock/bars?ticker=VCB&resolution=5&from=%d&to=%d&type=stock" % (ts[2024][0], ts[2024][1]))
test("TCBS v1 bars", "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker=VCB&type=stock&resolution=5&from=%d&to=%d" % (ts[2024][0], ts[2024][1]))
