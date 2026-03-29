"""
Converts raw fetcher dicts into clean markdown-style text blocks
for injection into the Claude synthesis prompt.
"""


def format_technical(data: dict) -> str:
    def fmt(v, prefix="", suffix="", decimals=2):
        if v is None:
            return "N/A"
        return f"{prefix}{v:.{decimals}f}{suffix}"

    lines = [
        f"Current Price:    {fmt(data['current_price'], '$')}",
        f"Trend:            {data['trend']}",
        f"",
        f"--- Momentum ---",
        f"RSI (14):         {fmt(data['rsi_14'])}",
        f"MACD:             {data['macd_signal']}",
        f"Stochastic %K/D:  {fmt(data['stoch_k'])} / {fmt(data['stoch_d'])}  -> {data['stoch_signal']}",
        f"",
        f"--- Trend ---",
        f"EMA 20:           {fmt(data['ema20'], '$')}",
        f"EMA 50:           {fmt(data['ema50'], '$')}",
        f"EMA 200:          {fmt(data['ema200'], '$')}",
        f"EMA Alignment:    {data['ema_alignment']}",
        f"ADX (14):         {fmt(data['adx_14'])}  ({'strong trend' if data['adx_14'] > 25 else 'weak trend'})",
        f"",
        f"--- Volatility ---",
        f"ATR (14):         {fmt(data['atr_14'], '$')}  ({fmt(data['atr_pct'], suffix='%')} of price)",
        f"Bollinger Bands:  {data['bb_position']}  (upper={fmt(data['bb_upper'], '$')}  lower={fmt(data['bb_lower'], '$')})",
        f"",
        f"--- Volume ---",
        f"Volume Trend:     {data['volume_trend']}",
        f"CMF (20):         {fmt(data['cmf_20'], decimals=4)}  -> {data['cmf_signal']}",
        f"",
        f"--- Key Levels ---",
        f"Nearest Support:  {fmt(data['nearest_support'], '$')}",
        f"Nearest Resist:   {fmt(data['nearest_resistance'], '$')}",
        f"52w High:         {fmt(data['high_52w'], '$')}",
        f"52w Low:          {fmt(data['low_52w'], '$')}",
    ]

    for level in data.get("key_levels", []):
        lines.append(f"  {level['label']:<14} {fmt(level['price'], '$')}  ({level['strength']})")

    return "\n".join(lines)


def format_macro(data: dict) -> str:
    def fmt(v, prefix="", suffix="", decimals=2):
        if v is None:
            return "N/A"
        return f"{prefix}{v:.{decimals}f}{suffix}"

    lines = [
        f"Fed Funds Rate:   {fmt(data['fed_funds_rate'], suffix='%')}",
        f"Yield Curve:      {data['yield_curve']}  (10Y-2Y spread: {fmt(data.get('yield_curve_spread'), suffix='%')})",
        f"CPI Trend:        {data['cpi_trend']}",
        f"Unemployment:     {fmt(data['unemployment_rate'], suffix='%')}",
        f"",
        f"VIX:              {fmt(data['vix_level'])}  -> {data['vix_regime']} volatility regime",
        f"USD Trend:        {data['usd_trend']}",
        f"Macro Bias:       {data['macro_bias']}",
    ]

    if data.get("wti_crude"):
        lines.append(f"WTI Crude:        {fmt(data['wti_crude'], '$')}/bbl")
    if data.get("gold_fix"):
        lines.append(f"Gold Fix:         {fmt(data['gold_fix'], '$')}/oz")

    if data.get("data_quality_flags"):
        lines.append(f"\nData Flags:       {', '.join(data['data_quality_flags'])}")

    return "\n".join(lines)


def format_news(articles: list[dict]) -> str:
    if not articles:
        return "No recent news available."

    lines = []
    for i, a in enumerate(articles, 1):
        pub = a.get("published_at", "")[:10]  # just the date
        lines.append(f"{i}. [{a['source']}] {a['title']}")
        if pub:
            lines.append(f"   Published: {pub}")
    return "\n".join(lines)


def format_geopolitical(data: dict) -> str:
    lines = [
        f"Risk Level:       {data['risk_level'].upper()}",
        f"",
        f"Key Factors:",
    ]
    for factor in data.get("key_factors", []):
        lines.append(f"  - {factor}")

    lines.append("")
    lines.append("Relevant Headlines:")
    for h in data.get("relevant_headlines", []):
        lines.append(f"  - {h}")

    return "\n".join(lines)
