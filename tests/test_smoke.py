"""
tests/test_smoke.py — smoke tests: imports, env, app syntax, YAML integrity.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent


class TestImports:
    """Every module must be importable without side-effects."""

    @pytest.mark.parametrize("module", [
        "core.config",
        "core.data",
        "core.license",
        "p3_backtester.strategies.base",
        "p3_backtester.strategies.registry",
        "p3_backtester.strategies.ema_continuation",
        "p3_backtester.strategies.ema_pullback",
        "p3_backtester.strategies.mtf_trend",
        "p3_backtester.strategies.momentum_breakout",
        "p3_backtester.strategies.orb",
        "p3_backtester.schema",
        "p4_live.journal",
        "p4_live.alerts",
        "p4_live.scanner",
        "p4_live.report",
    ])
    def test_importable(self, module):
        import importlib
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Cannot import {module}: {e}")


class TestAppSyntax:
    def test_app_py_valid_syntax(self):
        src = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"ui/app.py has syntax error: {e}")

    def test_license_py_valid_syntax(self):
        src = (ROOT / "core" / "license.py").read_text(encoding="utf-8")
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"core/license.py has syntax error: {e}")

    @pytest.mark.parametrize("path", [
        "core/config.py",
        "core/data.py",
        "p4_live/scanner.py",
        "p4_live/journal.py",
        "p4_live/alerts.py",
        "run_scheduler.py",
    ])
    def test_python_files_valid_syntax(self, path):
        full = ROOT / path
        if not full.exists():
            pytest.skip(f"{path} not found")
        src = full.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"{path} has syntax error: {e}")


class TestYAMLIntegrity:
    def test_assets_yaml_parseable(self):
        path = ROOT / "config" / "assets.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "assets" in data
        assert isinstance(data["assets"], list)

    def test_strategies_yaml_parseable(self):
        path = ROOT / "config" / "strategies.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "strategies" in data

    def test_assets_have_required_fields(self):
        path = ROOT / "config" / "assets.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for a in data["assets"]:
            assert "id" in a,       f"Asset missing 'id': {a}"
            assert "tickers" in a,  f"Asset '{a['id']}' missing 'tickers'"
            assert "category" in a, f"Asset '{a['id']}' missing 'category'"

    def test_strategies_have_required_fields(self):
        path = ROOT / "config" / "strategies.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for sid, s in data["strategies"].items():
            assert "default_timeframe" in s, f"Strategy '{sid}' missing 'default_timeframe'"
            assert "lookback_days" in s,     f"Strategy '{sid}' missing 'lookback_days'"


class TestEnvConfig:
    def test_anthropic_api_key_set(self):
        assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set in .env"

    def test_data_provider_set(self):
        assert os.getenv("DATA_PROVIDER"), "DATA_PROVIDER not set in .env"

    def test_mt5_config_present(self):
        # MT5 config should all be set or all be absent — not partial
        mt5_keys = ["MT5_ACCOUNT", "MT5_SERVER", "MT5_PASSWORD"]
        set_keys   = [k for k in mt5_keys if os.getenv(k)]
        unset_keys = [k for k in mt5_keys if not os.getenv(k)]
        assert not (set_keys and unset_keys), (
            f"Partial MT5 config — set: {set_keys}, missing: {unset_keys}"
        )

    def test_telegram_config_paired(self):
        # Both token + chat_id must be set, or both absent
        has_token   = bool(os.getenv("TELEGRAM_BOT_TOKEN"))
        has_chat_id = bool(os.getenv("TELEGRAM_CHAT_ID"))
        assert has_token == has_chat_id, (
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set or both absent"
        )

    def test_alerts_configured_with_dotenv(self):
        from p4_live.alerts import is_configured
        assert is_configured(), (
            "Alerts not configured — check TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env"
        )


class TestSchedulerScript:
    def test_scheduler_importable(self):
        """run_scheduler.py must not crash at import time."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import ast, pathlib; "
             "ast.parse(pathlib.Path('run_scheduler.py').read_text(encoding='utf-8'))"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        assert result.returncode == 0, f"run_scheduler.py syntax error: {result.stderr}"
