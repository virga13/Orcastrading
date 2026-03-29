import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="p1_analysis_engine",
        description="Orcastrading P1 — Asset Analysis Engine",
    )
    parser.add_argument("ticker", type=str, nargs="?", default=None, help="Asset ticker (e.g. AAPL, BTC-USD, EURUSD=X, GC=F)")
    parser.add_argument("--timeframe", type=str, default=None, help="Candlestick interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk (default: 1d)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save JSON output (default: outputs/)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving JSON file")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6", help="Claude model to use for synthesis")
    parser.add_argument("--json", action="store_true", help="Print raw JSON to stdout instead of rich display")
    parser.add_argument("--report", action="store_true", help="Generate HTML report and open in browser")
    parser.add_argument("--extended-thinking", action="store_true", help="Enable Claude extended thinking for deeper analysis (slower, uses more tokens)")
    return parser


def bias_color(bias: str) -> str:
    return {"bullish": "green", "bearish": "red", "neutral": "yellow"}.get(bias, "white")


def strength_color(strength: str) -> str:
    return {"strong": "bold", "moderate": "", "weak": "dim"}.get(strength, "")


def print_rich(bias) -> None:
    b_color = bias_color(bias.directional_bias)
    header = Text()
    header.append(f"  {bias.asset}", style="bold white")
    header.append(f"  ({bias.asset_class})", style="dim")
    header.append(f"  {bias.directional_bias.upper()}", style=f"bold {b_color}")
    header.append(f" {bias.bias_strength}", style=f"dim {b_color}")
    header.append(f"  confidence: {bias.confidence_score:.0%}", style="white")

    console.print()
    console.print(Panel(header, title="[bold]Orcastrading P1 — Asset Analysis Engine[/bold]", box=box.ROUNDED))

    # Core signal
    signal_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    signal_table.add_column(style="dim")
    signal_table.add_column()
    signal_table.add_row("Thesis", f"[italic]{bias.primary_thesis}[/italic]")
    signal_table.add_row("Timeframe", bias.suggested_timeframe)
    signal_table.add_row("Support", f"${bias.nearest_support:,.4f}")
    signal_table.add_row("Resistance", f"${bias.nearest_resistance:,.4f}")
    console.print(signal_table)

    # Bull / Bear
    cases = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 2))
    cases.add_column("[green]Bull Case[/green]", style="green", ratio=1)
    cases.add_column("[red]Bear Case[/red]", style="red", ratio=1)
    cases.add_row(bias.bull_case, bias.bear_case)
    console.print(cases)

    # Key risks
    console.print("\n[bold]Key Risks[/bold]")
    for i, risk in enumerate(bias.key_risks, 1):
        console.print(f"  [dim]{i}.[/dim] {risk}")

    # Technical snapshot
    t = bias.technical
    tech_table = Table(title="Technical Snapshot", box=box.SIMPLE, show_header=False, padding=(0, 2))
    tech_table.add_column(style="dim", width=20)
    tech_table.add_column()
    tech_table.add_row("Price", f"${t.current_price:,.4f}")
    tech_table.add_row("Trend", t.trend)
    tech_table.add_row("RSI (14)", str(t.rsi_14))
    tech_table.add_row("MACD", t.macd_signal)
    tech_table.add_row("EMA Alignment", t.ema_alignment)
    tech_table.add_row("ADX (14)", str(t.adx_14))
    tech_table.add_row("ATR (14)", f"${t.atr_14:,.4f}  ({t.atr_pct}%)")
    tech_table.add_row("Stoch %K/%D", f"{t.stoch_k} / {t.stoch_d}  ({t.stoch_signal})")
    tech_table.add_row("CMF (20)", f"{t.cmf_20}  ({t.cmf_signal})")
    tech_table.add_row("BB Position", t.bb_position)
    tech_table.add_row("Volume", t.volume_trend)
    console.print(tech_table)

    # Macro snapshot
    m = bias.macro
    macro_table = Table(title="Macro Snapshot", box=box.SIMPLE, show_header=False, padding=(0, 2))
    macro_table.add_column(style="dim", width=20)
    macro_table.add_column()
    macro_table.add_row("Fed Funds", f"{m.fed_funds_rate}%")
    macro_table.add_row("Yield Curve", m.yield_curve)
    macro_table.add_row("CPI Trend", m.cpi_trend)
    macro_table.add_row("VIX", f"{m.vix_level}  ({m.vix_regime})")
    macro_table.add_row("USD Trend", m.usd_trend)
    macro_table.add_row("Macro Bias", m.macro_bias)
    console.print(macro_table)

    # Recent news
    console.print("\n[bold]Recent News[/bold]")
    for item in bias.recent_news[:5]:
        pub = item.published_at[:10] if item.published_at else ""
        console.print(f"  [dim]{pub}[/dim]  [{item.source}] {item.title}")

    # Geopolitical
    g = bias.geopolitical
    geo_color = {"low": "green", "moderate": "yellow", "high": "red", "extreme": "bold red"}.get(g.risk_level, "white")
    console.print(f"\n[bold]Geopolitical Risk:[/bold] [{geo_color}]{g.risk_level.upper()}[/{geo_color}]")
    for h in g.relevant_headlines[:3]:
        console.print(f"  [dim]-[/dim] {h}")

    # Data quality flags
    if bias.data_quality_flags:
        console.print("\n[bold yellow]Data Quality Flags[/bold yellow]")
        for flag in bias.data_quality_flags:
            console.print(f"  [yellow dim]! {flag}[/yellow dim]")

    console.print(f"\n[dim]Analysis timestamp: {bias.analysis_timestamp}[/dim]\n")


INTERVAL_CHOICES = {
    "1": ("1m",  "1 min    — scalping, ultra short-term"),
    "2": ("5m",  "5 min    — scalping / intraday"),
    "3": ("15m", "15 min   — intraday / short swing"),
    "4": ("30m", "30 min   — intraday"),
    "5": ("1h",  "1 hour   — swing / intraday"),
    "6": ("1d",  "Daily    — swing / positional"),
    "7": ("1wk", "Weekly   — macro / positional"),
}


def _interactive_prompt() -> tuple[str, str]:
    """Show a rich input dialog when no ticker is provided."""
    from rich.prompt import Prompt

    console.print()
    console.print(Panel(
        "[bold gold1]ORCASTRADING[/bold gold1]  [dim]Asset Analysis Engine[/dim]",
        box=box.ROUNDED, padding=(1, 4)
    ))
    console.print()

    ticker = Prompt.ask("[bold]  Asset ticker[/bold]  [dim](e.g. AAPL, BTC, SILVER, EURUSD, GC=F)[/dim]")

    console.print()
    console.print("  [bold]Candlestick Timeframe[/bold]")
    for k, (interval, label) in INTERVAL_CHOICES.items():
        console.print(f"    [dim]{k}[/dim]  [bold]{interval:5}[/bold]  {label}")
    console.print()

    choice = Prompt.ask("  Select", choices=list(INTERVAL_CHOICES.keys()), default="6")
    interval, _ = INTERVAL_CHOICES[choice]

    console.print()
    return ticker.upper().strip(), interval


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Interactive mode when no ticker supplied
    if args.ticker is None:
        ticker, interval = _interactive_prompt()
        # Force --report on interactive runs for the richest output
        args.report = True
    else:
        ticker = args.ticker.upper().strip()
        interval = args.timeframe or "1d"

    extended_thinking = getattr(args, "extended_thinking", False)
    status_msg = f"[bold]Analyzing {ticker}  [{interval}]{'  (extended thinking)' if extended_thinking else ''}...[/bold]"

    with console.status(status_msg, spinner="dots"):
        if args.report:
            # Single merged Claude call — bias + setups in one round-trip with caching
            from p1_analysis_engine.engine import analyze_full
            try:
                bias, setups = analyze_full(ticker, model=args.model, interval=interval, extended_thinking=extended_thinking)
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                sys.exit(1)
        else:
            # Bias-only call (cheaper — no setups needed)
            from p1_analysis_engine.engine import analyze
            try:
                bias = analyze(ticker, model=args.model, interval=interval, extended_thinking=extended_thinking)
                setups = None
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                sys.exit(1)

    # Output
    if args.json:
        print(bias.model_dump_json(indent=2))
        return

    print_rich(bias)

    # HTML report
    if args.report:
        from p1_analysis_engine.utils.report import save_report
        import webbrowser
        report_path = save_report(bias, setups, args.output_dir, interval=interval)
        console.print(f"[dim]Report saved to {report_path}[/dim]")
        webbrowser.open(f"file:///{report_path}")

    # Save JSON
    if not args.no_save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"{ticker.replace('=', '').replace('-', '')}_{ts}.json"
        out_file.write_text(bias.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[dim]Saved to {out_file}[/dim]")


if __name__ == "__main__":
    main()
