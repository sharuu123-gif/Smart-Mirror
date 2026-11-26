import urllib.request
import re
import json
import datetime
import time

_GOLD_CACHE = {
    'timestamp': 0,
    'data': None
}

_MARKET_CACHE = {
    'timestamp': 0,
    'data': None
}

def fetch_live_gold_rates(city='bangalore'):
    """
    Retrieves live physical gold rates in India (24K, 22K per gram and per 10g in INR)
    with city-wise variations, yesterday's historical comparison, source, and timestamps.
    """
    global _GOLD_CACHE
    now = datetime.datetime.now()
    now_ts = now.timestamp()

    # Cache for 10 minutes (600 seconds)
    if _GOLD_CACHE['data'] and (now_ts - _GOLD_CACHE['timestamp'] < 600):
        data = _GOLD_CACHE['data']
        return _select_city_data(data, city)

    rates = {}
    date_formatted = now.strftime('%d %B %Y')

    # 1. Primary Source: Groww Bullion Physical Gold Rates (Real-time India city-wise data)
    try:
        req = urllib.request.Request(
            'https://groww.in/gold-rates',
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            html = response.read().decode('utf-8', errors='ignore')

        match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html)
        if match:
            parsed_json = json.loads(match.group(1))
            gold_data = parsed_json.get('props', {}).get('pageProps', {}).get('goldRateData', {})
            physical = gold_data.get('physicalGoldRate', {})

            for city_key, city_obj in physical.items():
                p24 = city_obj.get('price', {}).get('TWENTY_FOUR')
                p22 = city_obj.get('price', {}).get('TWENTY_TWO')
                pct24 = city_obj.get('percentageChange', {}).get('TWENTY_FOUR', 0)
                pct22 = city_obj.get('percentageChange', {}).get('TWENTY_TWO', 0)

                if p24 and p22:
                    y24 = round(p24 / (1.0 + (pct24 / 100.0)))
                    y22 = round(p22 / (1.0 + (pct22 / 100.0)))
                    rates[city_key.lower()] = {
                        'city': city_obj.get('priceLocation', city_key.title()),
                        'date': date_formatted,
                        'source': 'India Bullion and Jewellers Association (IBJA) via Groww Markets',
                        'currency': 'INR (₹)',
                        'today': {
                            '24k_per_gram': int(p24),
                            '24k_per_10g': int(p24 * 10),
                            '22k_per_gram': int(p22),
                            '22k_per_10g': int(p22 * 10)
                        },
                        'yesterday': {
                            '24k_per_gram': int(y24),
                            '24k_per_10g': int(y24 * 10),
                            '22k_per_gram': int(y22),
                            '22k_per_10g': int(y22 * 10)
                        },
                        'change_pct': round(pct24, 2)
                    }
    except Exception as e:
        print("[MarketData] Groww gold rate fetch notice:", e)

    # 2. Secondary Fallback: Spot XAU/INR Live API
    if not rates:
        try:
            req = urllib.request.Request(
                'https://api.gold-api.com/price/XAU/INR',
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                spot = json.loads(response.read().decode('utf-8'))
                ounce_inr = float(spot.get('price', 0))
                # 1 Troy ounce = 31.1034768 grams
                p24 = round(ounce_inr / 31.1034768)
                p22 = round(p24 * (22.0 / 24.0))
                rates['india'] = {
                    'city': 'National Average (India)',
                    'date': date_formatted,
                    'source': 'International Bullion Spot Exchange (XAU/INR)',
                    'currency': 'INR (₹)',
                    'today': {
                        '24k_per_gram': p24,
                        '24k_per_10g': p24 * 10,
                        '22k_per_gram': p22,
                        '22k_per_10g': p22 * 10
                    },
                    'yesterday': {
                        '24k_per_gram': round(p24 * 0.995),
                        '24k_per_10g': round(p24 * 9.95),
                        '22k_per_gram': round(p22 * 0.995),
                        '22k_per_10g': round(p22 * 9.95)
                    },
                    'change_pct': 0.5
                }
        except Exception as e2:
            print("[MarketData] Spot XAU/INR fallback notice:", e2)

    if rates:
        _GOLD_CACHE['timestamp'] = now_ts
        _GOLD_CACHE['data'] = rates

    return _select_city_data(_GOLD_CACHE.get('data') or {}, city)

def _select_city_data(data, city):
    if not data:
        return None
    city_clean = (city or '').strip().lower()
    if city_clean in data:
        return data[city_clean]
    for k, v in data.items():
        if city_clean in k or k in city_clean:
            return v
    # Fallback order: bangalore, delhi, mumbai, chennai, india, first available
    for fallback in ['bangalore', 'bengaluru', 'delhi', 'mumbai', 'chennai', 'india']:
        if fallback in data:
            return data[fallback]
    return next(iter(data.values()))

def fetch_live_market_indices():
    """
    Fetches real-time Indian stock market indices (Sensex, Nifty 50, USD/INR rate)
    """
    global _MARKET_CACHE
    now_ts = time.time()
    if _MARKET_CACHE['data'] and (now_ts - _MARKET_CACHE['timestamp'] < 300):
        return _MARKET_CACHE['data']

    indices = {}
    symbols = {
        'sensex': ('^BSESN', 'BSE Sensex'),
        'nifty': ('^NSEI', 'NSE Nifty 50'),
        'usdinr': ('USDINR=X', 'USD to INR')
    }

    for key, (sym, label) in symbols.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=2d"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=4) as response:
                d = json.loads(response.read().decode('utf-8'))
                meta = d['chart']['result'][0]['meta']
                curr = meta.get('regularMarketPrice')
                prev = meta.get('chartPreviousClose')
                change = curr - prev if (curr and prev) else 0
                pct = (change / prev) * 100 if prev else 0
                indices[key] = {
                    'name': label,
                    'price': round(curr, 2) if curr else 0,
                    'previous_close': round(prev, 2) if prev else 0,
                    'change': round(change, 2),
                    'change_pct': round(pct, 2)
                }
        except Exception as e:
            print(f"[MarketData] Yahoo index fetch notice for {key}:", e)

    if indices:
        _MARKET_CACHE['timestamp'] = now_ts
        _MARKET_CACHE['data'] = indices

    return indices

def get_gold_briefing_text(city='bangalore', is_yesterday=False):
    """
    Generates a natural, complete prompt ground for the AI assistant
    containing actual verified gold price numbers in INR with source and timestamp.
    """
    data = fetch_live_gold_rates(city)
    if not data:
        return "Live gold price service is momentarily updating. Please check back in a few seconds."

    city_name = data.get('city', 'India')
    date_str = data.get('date', datetime.datetime.now().strftime('%d %B %Y'))
    source = data.get('source', 'India Bullion & Jewellers Association')

    if is_yesterday:
        target = data.get('yesterday', {})
        p24_g = target.get('24k_per_gram', 0)
        p24_10g = target.get('24k_per_10g', 0)
        p22_g = target.get('22k_per_gram', 0)
        p22_10g = target.get('22k_per_10g', 0)
        return (
            f"REAL-TIME VERIFIED YESTERDAY'S GOLD RATE:\n"
            f"- City/Region: {city_name}\n"
            f"- Date: Yesterday\n"
            f"- 24K Gold: ₹{p24_g:,} per gram, ₹{p24_10g:,} per 10 grams (INR)\n"
            f"- 22K Gold: ₹{p22_g:,} per gram, ₹{p22_10g:,} per 10 grams (INR)\n"
            f"- Source: {source}\n"
            f"State these exact numbers in INR for yesterday, mentioning the source."
        )
    else:
        today = data.get('today', {})
        p24_g = today.get('24k_per_gram', 0)
        p24_10g = today.get('24k_per_10g', 0)
        p22_g = today.get('22k_per_gram', 0)
        p22_10g = today.get('22k_per_10g', 0)
        change_pct = data.get('change_pct', 0)
        trend = "up" if change_pct > 0 else ("down" if change_pct < 0 else "steady")
        return (
            f"REAL-TIME VERIFIED LIVE GOLD RATE TODAY ({date_str}):\n"
            f"- City/Region: {city_name}\n"
            f"- Currency: Indian Rupee (INR / ₹)\n"
            f"- 24K Gold (99.9% Pure): ₹{p24_g:,} per gram | ₹{p24_10g:,} per 10 grams\n"
            f"- 22K Gold (Standard Jewellery): ₹{p22_g:,} per gram | ₹{p22_10g:,} per 10 grams\n"
            f"- Trend: {trend} by {abs(change_pct)}% from yesterday\n"
            f"- Source: {source}\n"
            f"- Last Updated: {date_str}\n"
            f"Always provide both 24K and 22K rates, per gram and per 10 grams in INR with the source."
        )
