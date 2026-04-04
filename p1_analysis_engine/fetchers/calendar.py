"""
p1_analysis_engine/fetchers/calendar.py — High-impact economic event filter.

Signals fired on or adjacent to major macro events (FOMC, NFP, CPI) have
systematically worse outcomes because price discovery is disrupted by the release.
This module identifies those dates so the scanner can flag or skip them.

Events tracked:
  FOMC  — Federal Reserve rate decisions (8 per year, dates hardcoded annually)
  NFP   — Non-Farm Payrolls (first Friday of every month, released 08:30 ET)
  CPI   — Consumer Price Index (variable, ~2nd–3rd Tuesday; estimated here)

Usage:
    from p1_analysis_engine.fetchers.calendar import get_event_today

    event = get_event_today()
    if event:
        print(f"High-impact event today: {event['name']} — consider skipping signals")
"""
from __future__ import annotations
from datetime import date, timedelta


# ── FOMC decision dates (2025–2026) ──────────────────────────────────────────
# Second day of each two-day meeting = decision + press conference day.
# Update annually. Source: federalreserve.gov
_FOMC_DATES: set[date] = {
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
    # 2026
    date(2026, 1, 29),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 10),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
}


def _first_friday(year: int, month: int) -> date:
    """Return the first Friday of the given month."""
    d = date(year, month, 1)
    days_until_friday = (4 - d.weekday()) % 7   # Friday = weekday 4
    return d + timedelta(days=days_until_friday)


def _estimated_cpi_date(year: int, month: int) -> date:
    """
    CPI is typically released on the 2nd or 3rd Tuesday/Wednesday of the month.
    We use the 2nd Wednesday as a conservative estimate.
    Not exact — used only as a buffer window, not a precise filter.
    """
    d = date(year, month, 1)
    # Find first Wednesday (weekday 2)
    days_to_wed = (2 - d.weekday()) % 7
    first_wed   = d + timedelta(days=days_to_wed)
    return first_wed + timedelta(weeks=1)  # second Wednesday


def get_event_today(as_of: date | None = None, buffer_days: int = 1) -> dict | None:
    """
    Check if today (or a date within buffer_days) is a high-impact event.

    buffer_days=1 means: flag the day before AND the day of the event.
    This protects against positioning into a known catalyst.

    Returns None if no event, or a dict:
        {"name": "FOMC", "date": date(2026, 4, 29), "days_away": 0}

    Edge cases:
      - buffer_days=0 → only the exact event date is flagged
      - Returns the soonest event if multiple fall within the window
    """
    today = as_of or date.today()

    candidates = []

    # Check FOMC
    for fomc_date in _FOMC_DATES:
        days_away = (fomc_date - today).days
        if 0 <= days_away <= buffer_days or days_away == 0:
            candidates.append({"name": "FOMC", "date": fomc_date, "days_away": days_away})

    # Check NFP (first Friday of each month)
    for month_offset in [-1, 0, 1]:
        year  = today.year + (today.month + month_offset - 1) // 12
        month = (today.month + month_offset - 1) % 12 + 1
        nfp   = _first_friday(year, month)
        days_away = (nfp - today).days
        if 0 <= days_away <= buffer_days:
            candidates.append({"name": "NFP", "date": nfp, "days_away": days_away})

    # Check estimated CPI window
    for month_offset in [-1, 0, 1]:
        year  = today.year + (today.month + month_offset - 1) // 12
        month = (today.month + month_offset - 1) % 12 + 1
        cpi   = _estimated_cpi_date(year, month)
        days_away = (cpi - today).days
        if 0 <= days_away <= buffer_days:
            candidates.append({"name": "CPI", "date": cpi, "days_away": days_away})

    if not candidates:
        return None

    # Return the soonest event
    return min(candidates, key=lambda x: x["days_away"])
