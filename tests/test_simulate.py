"""
tests/test_simulate.py — Unit tests for p4_live.simulate core logic.

Tests cover _find_fill, _find_exit, and _simulate_trade edge cases:
  - fill within/outside timeout
  - slippage applied correctly for long and short
  - SL / TP1 / TP2 hit logic
  - partial wins (TP1 hit then SL)
  - data_exhausted returns None (not a forced close)
  - short direction throughout
"""
from __future__ import annotations

import pytest
import pandas as pd

from p4_live.simulate import _find_fill, _find_exit, _simulate_trade, _ENTRY_TIMEOUT


def _make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from (open, high, low, close) tuples."""
    dates = pd.date_range("2024-01-01", periods=len(bars), freq="D")
    return pd.DataFrame(
        [{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1_000.0}
         for o, h, l, c in bars],
        index=dates,
    )


# ── _find_fill ─────────────────────────────────────────────────────────────────

class TestFindFill:
    def test_long_fills_bar1(self):
        df = _make_df([
            (100, 102, 98, 100),   # bar 0 — signal bar, must be skipped
            (100, 105, 99, 103),   # bar 1 — High 105 >= entry_high 103
        ])
        result = _find_fill(df, "long", entry_low=101, entry_high=103)
        assert result is not None
        fill_price, fill_idx = result
        assert fill_idx == 1
        assert fill_price == pytest.approx(103.0)   # min(entry_high, bar_high)

    def test_long_skips_signal_bar(self):
        # Zone is satisfied on bar 0 — must not fill on the signal bar itself
        df = _make_df([
            (100, 105, 99, 103),   # would fill if checked
        ])
        assert _find_fill(df, "long", entry_low=101, entry_high=103) is None

    def test_long_no_fill_when_price_above_zone(self):
        # Price stays above the buy zone (never pulls back)
        bars = [(200, 210, 195, 200)] * (_ENTRY_TIMEOUT + 2)
        df = _make_df(bars)
        assert _find_fill(df, "long", entry_low=100, entry_high=110) is None

    def test_long_fills_at_last_allowed_bar(self):
        # Zone hit exactly at bar _ENTRY_TIMEOUT (last bar allowed by the window)
        bars = [(200, 210, 195, 200)] * _ENTRY_TIMEOUT
        bars.append((100, 105, 99, 103))   # bar _ENTRY_TIMEOUT — within window
        df = _make_df(bars)
        result = _find_fill(df, "long", entry_low=101, entry_high=103)
        assert result is not None
        _, fill_idx = result
        assert fill_idx == _ENTRY_TIMEOUT

    def test_long_no_fill_after_timeout(self):
        # Zone hit at bar _ENTRY_TIMEOUT + 1 — outside the allowed window
        bars = [(200, 210, 195, 200)] * (_ENTRY_TIMEOUT + 1)
        bars.append((100, 105, 99, 103))
        df = _make_df(bars)
        assert _find_fill(df, "long", entry_low=101, entry_high=103) is None

    def test_long_slippage_worsens_fill(self):
        df = _make_df([
            (100, 102, 98, 100),
            (100, 105, 99, 103),
        ])
        result = _find_fill(df, "long", entry_low=101, entry_high=103, slippage_pct=1.0)
        assert result is not None
        fill_price, _ = result
        assert fill_price == pytest.approx(103.0 * 1.01)

    def test_short_fills_bar1(self):
        # Short: High >= entry_low and Low <= entry_high
        df = _make_df([
            (100, 102, 98, 100),   # signal bar
            (100, 102, 97, 99),    # Low 97 <= entry_high 100; High 102 >= entry_low 98
        ])
        result = _find_fill(df, "short", entry_low=98, entry_high=100)
        assert result is not None
        fill_price, fill_idx = result
        assert fill_idx == 1
        assert fill_price == pytest.approx(98.0)   # max(entry_low, bar_low)

    def test_short_slippage_worsens_fill(self):
        df = _make_df([
            (100, 102, 98, 100),
            (100, 102, 97, 99),
        ])
        result = _find_fill(df, "short", entry_low=98, entry_high=100, slippage_pct=1.0)
        assert result is not None
        fill_price, _ = result
        assert fill_price == pytest.approx(98.0 * 0.99)

    def test_short_no_fill_when_price_below_zone(self):
        bars = [(50, 60, 45, 52)] * (_ENTRY_TIMEOUT + 2)
        df = _make_df(bars)
        assert _find_fill(df, "short", entry_low=80, entry_high=85) is None


# ── _find_exit ─────────────────────────────────────────────────────────────────

class TestFindExit:
    def test_long_sl_hit(self):
        df = _make_df([
            (100, 105, 99, 102),   # fill bar (bar 0, not checked)
            (98,  99,  92, 93),    # Low 92 <= SL 95
        ])
        result = _find_exit(df, "long", fill_price=100.0, stop_loss=95.0,
                            tp1=115.0, tp2=130.0, tp1_alloc=70, tp2_alloc=30)
        assert result is not None
        assert result["outcome"] == "loss"
        assert result["exit_price"] == pytest.approx(95.0)
        assert result["pnl_r"] < 0

    def test_long_tp1_and_tp2_win(self):
        df = _make_df([
            (100, 102, 99, 101),   # fill bar
            (101, 116, 100, 115),  # High 116 >= TP1 115
            (115, 132, 114, 131),  # High 132 >= TP2 130
        ])
        result = _find_exit(df, "long", fill_price=100.0, stop_loss=95.0,
                            tp1=115.0, tp2=130.0, tp1_alloc=70, tp2_alloc=30)
        assert result is not None
        assert result["outcome"] == "win"
        assert result["pnl_r"] > 0

    def test_long_partial_win_tp1_then_sl(self):
        # TP1 hit (70%), then SL hit on remaining 30% — net positive → partial_win
        df = _make_df([
            (100, 102, 99, 101),   # fill bar
            (101, 116, 100, 115),  # TP1 hit at 115 (+15 pts × 70%)
            (115, 116, 91, 92),    # SL hit at 95 (-5 pts × 30%)
        ])
        result = _find_exit(df, "long", fill_price=100.0, stop_loss=95.0,
                            tp1=115.0, tp2=130.0, tp1_alloc=70, tp2_alloc=30)
        assert result is not None
        assert result["outcome"] in ("partial_win", "win")
        assert result["pnl_r"] > 0

    def test_long_data_exhausted_returns_none(self):
        # No SL or TP hit — must return None (not a forced close)
        df = _make_df([
            (100, 102, 99, 101),   # fill bar
            (101, 103, 99, 102),   # no hit
            (102, 104, 100, 103),  # no hit
        ])
        result = _find_exit(df, "long", fill_price=100.0, stop_loss=95.0,
                            tp1=115.0, tp2=130.0, tp1_alloc=70, tp2_alloc=30)
        assert result is None

    def test_long_fill_bar_not_checked_for_exit(self):
        # SL at 95, fill bar Low = 90 — must NOT trigger on fill bar (bar 0)
        df = _make_df([
            (100, 102, 90, 101),   # fill bar — Low breaches SL, but bar 0 is skipped
            (101, 103, 99, 102),   # no hit
        ])
        result = _find_exit(df, "long", fill_price=100.0, stop_loss=95.0,
                            tp1=115.0, tp2=None, tp1_alloc=100, tp2_alloc=0)
        assert result is None   # data exhausted, not SL

    def test_short_sl_hit(self):
        df = _make_df([
            (100, 102, 98, 100),   # fill bar
            (100, 107, 99, 106),   # High 107 >= SL 105
        ])
        result = _find_exit(df, "short", fill_price=100.0, stop_loss=105.0,
                            tp1=90.0, tp2=80.0, tp1_alloc=70, tp2_alloc=30)
        assert result is not None
        assert result["outcome"] == "loss"
        assert result["pnl_r"] < 0

    def test_short_tp1_and_tp2_win(self):
        df = _make_df([
            (100, 102, 98, 100),   # fill bar
            (100, 101, 88, 89),    # Low 88 <= TP1 90
            (89,  90,  78, 79),    # Low 78 <= TP2 80
        ])
        result = _find_exit(df, "short", fill_price=100.0, stop_loss=105.0,
                            tp1=90.0, tp2=80.0, tp1_alloc=70, tp2_alloc=30)
        assert result is not None
        assert result["outcome"] == "win"
        assert result["pnl_r"] > 0

    def test_zero_risk_returns_none(self):
        # fill_price == stop_loss — degenerate, should not crash
        df = _make_df([
            (100, 102, 98, 100),
            (100, 112, 99, 111),
        ])
        result = _find_exit(df, "long", fill_price=100.0, stop_loss=100.0,
                            tp1=110.0, tp2=None, tp1_alloc=100, tp2_alloc=0)
        assert result is None

    def test_single_target_no_tp2(self):
        # tp2=None, tp2_alloc=0 — entire position closes at TP1
        df = _make_df([
            (100, 102, 99, 101),   # fill bar
            (101, 112, 100, 111),  # TP1 hit
        ])
        result = _find_exit(df, "long", fill_price=100.0, stop_loss=95.0,
                            tp1=110.0, tp2=None, tp1_alloc=100, tp2_alloc=0)
        assert result is not None
        assert result["outcome"] == "win"


# ── _simulate_trade ────────────────────────────────────────────────────────────

class TestSimulateTrade:
    def test_expired_returns_none(self):
        bars = [(200, 210, 195, 200)] * (_ENTRY_TIMEOUT + 5)
        df = _make_df(bars)
        result = _simulate_trade("long", 100, 110, 90, 130, 150, 70, 30, df)
        assert result is None

    def test_filled_data_exhausted_no_exit_fields(self):
        # Fills bar 1 but only 1 more bar — SL/TP not hit yet
        df = _make_df([
            (100, 102, 98, 100),   # signal bar
            (100, 105, 99, 103),   # fills at 103
            (103, 104, 101, 102),  # no SL/TP hit
        ])
        result = _simulate_trade("long", 101, 103, 95, 115, 125, 70, 30, df)
        assert result is not None
        assert result["data_exhausted"] is True
        assert "fill_price" in result
        assert "fill_date" in result
        assert "exit_price" not in result
        assert "outcome" not in result

    def test_filled_and_closed_win(self):
        df = _make_df([
            (100, 102, 98, 100),   # signal
            (100, 105, 99, 103),   # fill at 103
            (103, 116, 102, 115),  # TP1 110 hit
            (115, 128, 114, 127),  # TP2 120 hit
        ])
        result = _simulate_trade("long", 101, 103, 95, 110, 120, 70, 30, df)
        assert result is not None
        assert result["data_exhausted"] is False
        assert result["outcome"] == "win"
        assert result["pnl_r"] > 0

    def test_filled_and_closed_loss(self):
        df = _make_df([
            (100, 102, 98, 100),   # signal
            (100, 105, 99, 103),   # fill at 103
            (103, 104, 91, 92),    # SL 95 hit
        ])
        result = _simulate_trade("long", 101, 103, 95, 115, 125, 70, 30, df)
        assert result is not None
        assert result["data_exhausted"] is False
        assert result["outcome"] == "loss"
        assert result["pnl_r"] < 0

    def test_slippage_increases_long_fill_price(self):
        df = _make_df([
            (100, 102, 98, 100),
            (100, 105, 99, 103),
            (103, 104, 91, 92),
        ])
        r0 = _simulate_trade("long", 101, 103, 95, 115, None, 100, 0, df, slippage_pct=0.0)
        r1 = _simulate_trade("long", 101, 103, 95, 115, None, 100, 0, df, slippage_pct=1.0)
        assert r0 is not None and r1 is not None
        assert r1["fill_price"] > r0["fill_price"]

    def test_slippage_decreases_short_fill_price(self):
        df = _make_df([
            (100, 102, 98, 100),
            (100, 102, 97, 98),    # fills short
            (98,  99, 107, 108),   # SL hit
        ])
        r0 = _simulate_trade("short", 98, 100, 108, 88, None, 100, 0, df, slippage_pct=0.0)
        r1 = _simulate_trade("short", 98, 100, 108, 88, None, 100, 0, df, slippage_pct=1.0)
        assert r0 is not None and r1 is not None
        assert r1["fill_price"] < r0["fill_price"]

    def test_short_expired(self):
        bars = [(50, 60, 45, 52)] * (_ENTRY_TIMEOUT + 5)
        df = _make_df(bars)
        result = _simulate_trade("short", 80, 85, 95, 70, 60, 70, 30, df)
        assert result is None

    def test_short_win(self):
        df = _make_df([
            (100, 102, 98, 100),   # signal
            (100, 102, 97, 98),    # fill short at 98
            (98,  99,  87, 88),    # TP1 90 hit
            (88,  89,  78, 79),    # TP2 80 hit
        ])
        result = _simulate_trade("short", 98, 100, 108, 90, 80, 70, 30, df)
        assert result is not None
        assert result.get("outcome") == "win"
        assert result["pnl_r"] > 0

    def test_fill_date_matches_fill_bar(self):
        df = _make_df([
            (100, 102, 98, 100),
            (100, 105, 99, 103),   # fills on 2024-01-02
            (103, 104, 101, 102),
        ])
        result = _simulate_trade("long", 101, 103, 95, 115, None, 100, 0, df)
        assert result is not None
        assert result["fill_date"] == "2024-01-02"
