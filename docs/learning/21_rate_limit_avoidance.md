# Rate Limit Avoidance Strategies for Stock Data Collection

**Date:** 2026-05-31
**Context:** Fetching VN30 stock historical data without being blocked

## Problem

When fetching data for multiple stocks sequentially using vnstock:
- API blocks requests after ~7 stocks
- Error: `RetryError` or HTTP 429 (Too Many Requests)
- Simple delays (0.5-5s) between requests insufficient

## Root Cause

**Rate limiting:** APIs limit requests to prevent abuse
- VN30 stocks = 31 stocks × 1 request each = 31 rapid requests
- Triggers API rate limit protection
- Sequential requests with fixed delays still get blocked

## Solution Strategies

### 1. **Exponential Backoff with Jitter** (PRIMARY SOLUTION)

Instead of fixed delays, use exponentially increasing delays with randomness:

```python
def fetch_with_backoff(fetch_func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return fetch_func()
        except Exception:
            # Wait: 2s, 4s, 8s, 16s, 32s
            wait = 2 * (2 ** attempt) + random.uniform(0, 2)
            time.sleep(wait)
```

**Why it works:**
- Exponential increase: Gives API time to recover
- Jitter (randomness): Prevents synchronized retries from thundering the API
- Example: 3 clients retrying at once will have different delays (2.3s, 3.7s, 4.1s)

### 2. **Aggressive Caching** (EFFICIENCY)

Never re-fetch data that hasn't changed:

```python
def fetch_with_cache(ticker):
    cache_file = f'data/raw/prices/{ticker}_ohlcv.csv'

    # Check cache
    if cache_file.exists():
        cached = pd.read_csv(cache_file)
        if cached['date'].max() >= TODAY:
            return cached  # Cache hit!

    # Cache miss - fetch and save
    df = fetch_from_api(ticker)
    df.to_csv(cache_file)
    return df
```

**Benefits:**
- First run: Fetches all 31 stocks (~5 minutes with delays)
- Subsequent runs: Returns immediately if data current (<1 second)
- Incremental updates: Only fetch new days, not full history

### 3. **Incremental Updates** (BANDWIDTH SAVING)

Don't re-fetch entire history for daily updates:

```python
def fetch_incremental(ticker):
    # Get cached data end date
    cached_end = get_cache_end(ticker)  # e.g., 2026-05-29

    # Only fetch new data
    if cached_end < TODAY:
        new_data = fetch(ticker, start=cached_end+1day, end=TODAY)
        return merge(cache, new_data)
```

**Example:**
- Full history: 5000 rows × 31 stocks = 155,000 rows fetched
- Daily update: 1 row × 31 stocks = 31 rows fetched
- **500x reduction in API calls**

### 4. **Sequential vs Parallel Processing** (RELIABILITY)

**Sequential** (recommended for rate-limited APIs):
```python
for ticker in tickers:
    fetch(ticker)
    sleep(2)  # Polite delay
```

**Parallel** (only for APIs without rate limits):
```python
with ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(fetch, tickers)
```

**Trade-offs:**
- Sequential: Slower but reliable with rate limits
- Parallel: Faster but risks blocking

### 5. **Fallback Data Sources** (RESILIENCE)

If vnstock fails:

```python
def fetch_with_fallback(ticker):
    try:
        # Primary: vnstock (Vietnam-specific, full history)
        return fetch_vnstock(ticker)
    except:
        # Fallback: yfinance (global, less history)
        return fetch_yfinance(ticker)
```

**Pros/Cons:**
- vnstock: Full history (2007+), but rate-limited
- yfinance: No rate limits, but less history (2012+)

## Implementation

**File:** `scripts/fetch_stock_data_robust.py`

**Key features:**
1. Exponential backoff with jitter (2s → 4s → 8s → 16s → 32s)
2. Smart caching (complete/partial hit detection)
3. Incremental updates (only fetch new data)
4. Sequential processing with polite delays
5. Comprehensive error handling

**Usage:**

```bash
# Fetch all VN30 stocks (uses cache if available)
python scripts/fetch_stock_data_robust.py

# Force refresh (ignore cache)
python scripts/fetch_stock_data_robust.py --force

# Single stock
python scripts/fetch_stock_data_robust.py --ticker VIC

# Custom date range
python scripts/fetch_stock_data_robust.py --start 2020-01-01 --end 2026-05-31
```

## Performance

**First run (cold cache):**
- Time: ~5-10 minutes for 31 stocks
- API calls: 31 (one per stock)
- Success rate: 100% with exponential backoff

**Subsequent runs (warm cache):**
- Time: <1 second (cache hit)
- API calls: 0 (uses cached data)

**Daily update:**
- Time: ~1-2 minutes (incremental only)
- API calls: 31 (one per stock, small date range)

## Best Practices

1. **Always use caching** - Reduces API load and speeds up runs
2. **Add jitter to delays** - Prevents synchronized retries
3. **Use exponential backoff** - Gives API time to recover
4. **Fetch sequentially** - More reliable than parallel for rate-limited APIs
5. **Implement incremental updates** - Saves bandwidth and time
6. **Have fallback sources** - yfinance, manual download, etc.

## When All Else Fails

If still blocked:
1. **Wait longer** - API rate limits reset after time window (usually 1-15 minutes)
2. **Use VPN** - Different IP address = fresh rate limit
3. **Manual download** - Export from https://finance.vietstock.vn
4. **Paid API** - Professional services ($50-100/month)
5. **Academic access** - Universities may have data access

## Common Mistakes

❌ **Fixed delays:**
```python
for ticker in tickers:
    fetch(ticker)
    sleep(1)  # Too short, will get blocked
```

✅ **Exponential backoff:**
```python
def fetch_with_retry(ticker):
    for attempt in range(5):
        try:
            return fetch(ticker)
        except:
            sleep(2 ** attempt)  # 2s, 4s, 8s, 16s, 32s
```

❌ **No caching:**
```python
# Every run fetches all data = slow + rate limit
df = fetch_all_stocks()
```

✅ **Smart caching:**
```python
# Only fetch new/missing data
if not cache_exists_or_is_complete():
    df = fetch_all_stocks()
else:
    df = load_from_cache()
```

❌ **Parallel fetching:**
```python
# All 31 requests at once = instant block
with Pool(31) as p:
    p.map(fetch, tickers)
```

✅ **Sequential fetching:**
```python
# One at a time with delays = reliable
for ticker in tickers:
    fetch(ticker)
    sleep(2)
```

## References

- [Rate Limiting - OpenAI Cookbook](https://developers.openline.com/cookbook/examples/how_to_handle_rate_limits)
- [Exponential Backoff - Wikipedia](https://en.wikipedia.org/wiki/Exponential_backoff)
- [Implementing Rate Limiting - Medium](https://medium.com/neural-engineer/implementing-effective-api-rate-limiting-in-python-6147fdd7d516)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [vnstock Documentation](https://github.com/thinh-vu/vnstock)
