"""
Crawl VIC stock data - alternative methods
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("\n" + "="*70)
print("  VIC DATA CRAWLING - MULTIPLE METHODS")
print("="*70 + "\n")

DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'prices' / 'VIC_ohlcv.csv'

# Load existing data
existing_data = pd.read_csv(DATA_FILE)
existing_data['date'] = pd.to_datetime(existing_data['date'])

print(f"[Current Data]")
print(f"  Rows: {len(existing_data)}")
print(f"  Date range: {existing_data['date'].iloc[0].date()} to {existing_data['date'].iloc[-1].date()}")

# =============================================================================
# METHOD 1: Try vnstock (VCI source)
# =============================================================================

print(f"\n[Method 1] vnstock VCI source...")

try:
    from vnstock import Vnstock
    import pandas as pd

    vn = Vnstock()

    # Try listing function
    try:
        # Try to get historical data with listing function
        df = vn.stock(symbol='VIC', source='VCI', start='2026-05-01')

        if isinstance(df, pd.DataFrame) and len(df) > 0:
            print(f"  [OK] Fetched {len(df)} rows")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
        else:
            print(f"  [FAIL] Unexpected data type: {type(df)}")
    except Exception as e:
        print(f"  [FAIL] Error: {e}")

except Exception as e:
    print(f"  [FAIL] vnstock not available: {e}")

# =============================================================================
# METHOD 2: Try web crawling with requests
# =============================================================================

print(f"\n[Method 2] Web crawling...")

try:
    import requests
    from bs4 import BeautifulSoup

    # Vietstock Finance URL for VIC
    url = "https://finance.vietstock.vn/data/VIC/historical"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    response = requests.get(url, headers=headers, timeout=10)
    print(f"  Response status: {response.status_code}")

    if response.status_code == 200:
        try:
            data = response.json()
            print(f"  [OK] Got JSON data")

            if isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
                print(f"  Columns: {df.columns.tolist()}")
                print(f"  Rows: {len(df)}")
        except Exception as e:
            print(f"  [FAIL] JSON parse error: {e}")
    else:
        print(f"  [FAIL] HTTP {response.status_code}")

except Exception as e:
    print(f"  [FAIL] Web crawling failed: {e}")

# =============================================================================
# METHOD 3: Manual instructions
# =============================================================================

print(f"\n[Method 3] Manual download instructions")
print(f"  1. Visit: https://finance.vietstock.vn/VIC#hist")
print(f"  2. Select date range: from 2026-05-16 to present")
print(f"  3. Click 'Download'")
print(f"  4. Save as CSV")
print(f"  5. Place in: {DATA_FILE}")
print(f"  6. Required columns: date, open, high, low, close, volume")

print(f"\n{'='*70}\n")

# =============================================================================
# RECOMMENDATION
# =============================================================================

print(f"[Recommendation]")
print(f"  Current data already extends to {existing_data['date'].iloc[-1].date()}")
print(f"  May 2026 has {len(existing_data[existing_data['date'] >= '2026-05-01'])} close price days")
print(f"  ")
print(f"  For now, 5 May RV samples (ending May 8) is sufficient for:")
print(f"  - Testing pipeline validation")
print(f"  - Model training")
print(f"  - Learning curve visualization")
print(f"  ")
print(f"  You can crawl more data later when:")
print(f"  - More May/June data becomes available")
print(f"  - You want to extend test period")
print(f"  - You need longer backtesting period")

print(f"\n{'='*70}\n")
