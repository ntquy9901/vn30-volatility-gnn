"""Smoke test: CafeF chart, iTick, algotrade, Vietstock intraday APIs."""
import sys
import requests
import datetime

sys.stdout.reconfigure(line_buffering=True)

headers_browser = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def test_url(label, url, method="GET", payload=None, extra_headers=None, timeout=10):
    h = dict(headers_browser)
    if extra_headers:
        h.update(extra_headers)
    try:
        if method == "POST":
            r = requests.post(url, json=payload, headers=h, timeout=timeout)
        else:
            r = requests.get(url, headers=h, timeout=timeout)
        print("[%d] %s" % (r.status_code, label))
        if r.status_code == 200:
            try:
                d = r.json()
                if isinstance(d, list):
                    n = len(d)
                    print("    list len=%d" % n)
                    if n > 0 and isinstance(d[0], dict):
                        print("    keys=%s" % list(d[0].keys())[:8])
                elif isinstance(d, dict):
                    print("    keys=%s" % list(d.keys())[:8])
                    for k in ["t", "data", "bars", "candles", "items", "ohlc"]:
                        if k in d and d[k]:
                            arr = d[k]
                            n = len(arr) if hasattr(arr, "__len__") else "?"
                            print("    d['%s']: %s items" % (k, n))
                            if isinstance(arr, list) and len(arr) > 0:
                                print("    first=%s" % str(arr[0])[:100])
                            break
            except Exception as e:
                print("    json err: %s | body: %s" % (e, r.text[:100]))
        elif r.status_code in [401, 403]:
            print("    auth required")
        else:
            print("    body: %s" % r.text[:150])
    except Exception as e:
        print("[ERR] %s: %s" % (label, str(e)[:100]))


ts_2024 = int(datetime.datetime(2024, 1, 2, 9, 0).timestamp())
te_2024 = int(datetime.datetime(2024, 1, 6, 15, 0).timestamp())
ts_2020 = int(datetime.datetime(2020, 1, 2, 9, 0).timestamp())
te_2020 = int(datetime.datetime(2020, 1, 6, 15, 0).timestamp())
ts_2015 = int(datetime.datetime(2015, 1, 5, 9, 0).timestamp())
te_2015 = int(datetime.datetime(2015, 1, 9, 15, 0).timestamp())

print("=" * 70)
print("iTick API")
print("=" * 70)
test_url("iTick /stock/history 5m 2024", "https://api.itick.vn/stock/history?sym=VCB&resolution=5&from=%d&to=%d" % (ts_2024, te_2024))
test_url("iTick /candles 5m 2024", "https://api.itick.vn/candles?symbol=VCB&resolution=5&from=%d&to=%d" % (ts_2024, te_2024))
test_url("iTick /history 5m 2024", "https://api.itick.vn/history?symbol=VCB&resolution=5&from=%d&to=%d" % (ts_2024, te_2024))
test_url("iTick v2 /stock/history", "https://apiv2.itick.vn/stock/history?sym=VCB&resolution=5&from=%d&to=%d" % (ts_2024, te_2024))

print()
print("=" * 70)
print("AlgoTrade API")
print("=" * 70)
test_url("algotrade /history 5m", "https://api.algotrade.vn/data/stock/history?symbol=VCB&resolution=5&from=%d&to=%d" % (ts_2024, te_2024))
test_url("algotrade /ohlcv", "https://api.algotrade.vn/v1/ohlcv?symbol=VCB&interval=5m&start=2024-01-02&end=2024-01-06")

print()
print("=" * 70)
print("CafeF chart API")
print("=" * 70)
test_url("cafef chart 5m", "https://cafef.vn/Ajax/GetDataChart.aspx?symbol=VCB&type=intraday&date=02/01/2024",
         extra_headers={"Referer": "https://cafef.vn/"})
test_url("cafef price history", "https://cafef.vn/Ajax/AjaxStockPaging.aspx?symbol=VCB&type=0&pageIndex=1&pageSize=10&module=ThiTruong",
         extra_headers={"Referer": "https://cafef.vn/"})

print()
print("=" * 70)
print("SSI iboard chart API (correct endpoint)")
print("=" * 70)
# SSI iboard uses TradingView-compatible Datafeed API
test_url("SSI iboard history 5m 2024", "https://iboard.ssi.com.vn/dchart/api/history?resolution=5&symbol=VCB&from=%d&to=%d" % (ts_2024, te_2024),
         extra_headers={"Referer": "https://iboard.ssi.com.vn/"})
test_url("SSI iboard history D 2024", "https://iboard.ssi.com.vn/dchart/api/history?resolution=D&symbol=VCB&from=%d&to=%d" % (ts_2024, te_2024),
         extra_headers={"Referer": "https://iboard.ssi.com.vn/"})
test_url("SSI iboard history 5m 2020", "https://iboard.ssi.com.vn/dchart/api/history?resolution=5&symbol=VCB&from=%d&to=%d" % (ts_2020, te_2020),
         extra_headers={"Referer": "https://iboard.ssi.com.vn/"})
test_url("SSI iboard history 5m 2015", "https://iboard.ssi.com.vn/dchart/api/history?resolution=5&symbol=VCB&from=%d&to=%d" % (ts_2015, te_2015),
         extra_headers={"Referer": "https://iboard.ssi.com.vn/"})

print()
print("=" * 70)
print("Vietstock chart API")
print("=" * 70)
# Vietstock uses TradingView Datafeed
test_url("vietstock chart 5m 2024", "https://finance.vietstock.vn/chart/getDataChart.aspx?symbol=VCB&fiboresolution=5&from=2024-01-02&to=2024-01-06",
         extra_headers={"Referer": "https://vietstock.vn/", "Cookie": "vst_usr_lg_token=; ASP.NET_SessionId="})
test_url("vietstock history bar", "https://api.vietstock.vn/chart/history?symbol=VCB&resolution=5&from=%d&to=%d" % (ts_2024, te_2024),
         extra_headers={"Referer": "https://vietstock.vn/"})
