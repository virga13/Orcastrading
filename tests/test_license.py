"""
tests/test_license.py — unit tests for core/license.py
"""
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestMachineId:
    def test_returns_string(self):
        from core.license import get_machine_id
        mid = get_machine_id()
        assert isinstance(mid, str)
        assert len(mid) >= 8

    def test_stable(self):
        from core.license import get_machine_id
        assert get_machine_id() == get_machine_id()

    def test_fallback_without_machineid_package(self):
        """Should not crash even if py-machineid isn't installed."""
        from core import license as lic
        with patch.dict("sys.modules", {"machineid": None}):
            mid = lic.get_machine_id()
            assert isinstance(mid, str) and len(mid) > 0


class TestMaskKey:
    def test_masks_prefix(self):
        from core.license import _mask_key
        assert _mask_key("ABCD-EFGH-IJKL-MNOP") == "****-****-****-MNOP"

    def test_single_segment(self):
        from core.license import _mask_key
        result = _mask_key("ABCDEFGH")
        assert result.endswith("EFGH") or "****" in result

    def test_empty_key(self):
        from core.license import _mask_key
        result = _mask_key("")
        assert isinstance(result, str)


class TestDevBypass:
    def test_check_returns_true_in_dev_mode(self):
        from core import license as lic
        with patch.dict(os.environ, {"ORCA_DEV_MODE": "1"}):
            ok, msg = lic.check()
        assert ok is True
        assert msg == "dev"

    @pytest.mark.parametrize("val", ["1", "true", "True", "yes", "YES"])
    def test_dev_mode_variants(self, val):
        from core import license as lic
        with patch.dict(os.environ, {"ORCA_DEV_MODE": val}):
            ok, _ = lic.check()
        assert ok is True


class TestCacheOperations:
    def test_load_cache_returns_none_when_missing(self, tmp_path, monkeypatch):
        from core import license as lic
        monkeypatch.setattr(lic, "_CACHE_FILE", tmp_path / "nonexistent.json")
        assert lic._load_cache() is None

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        from core import license as lic
        monkeypatch.setattr(lic, "_CACHE_DIR",  tmp_path)
        monkeypatch.setattr(lic, "_CACHE_FILE", tmp_path / "license.json")
        data = {
            "license_key": "TEST-KEY",
            "instance_id": "inst-001",
            "machine_id":  "abc123",
            "validated_at": "2026-01-01T00:00:00+00:00",
            "expires_grace": "2026-01-08T00:00:00+00:00",
            "customer_email": "test@example.com",
            "product_name": "Orcastrading",
        }
        lic._save_cache(data)
        loaded = lic._load_cache()
        assert loaded["license_key"] == "TEST-KEY"
        assert loaded["customer_email"] == "test@example.com"

    def test_clear_cache(self, tmp_path, monkeypatch):
        from core import license as lic
        cache_file = tmp_path / "license.json"
        cache_file.write_text('{"license_key":"X"}')
        monkeypatch.setattr(lic, "_CACHE_FILE", cache_file)
        lic._clear_cache()
        assert not cache_file.exists()


class TestCheckWithoutNetwork:
    def test_no_cache_returns_false(self, tmp_path, monkeypatch):
        from core import license as lic
        monkeypatch.setattr(lic, "_CACHE_FILE", tmp_path / "no.json")
        with patch.dict(os.environ, {"ORCA_DEV_MODE": "0"}):
            ok, msg = lic.check()
        assert ok is False
        assert msg == "no_license"

    def test_wrong_machine_returns_false(self, tmp_path, monkeypatch):
        from core import license as lic
        monkeypatch.setattr(lic, "_CACHE_DIR",  tmp_path)
        monkeypatch.setattr(lic, "_CACHE_FILE", tmp_path / "license.json")
        # Write cache with a different machine_id
        lic._save_cache({
            "license_key": "XXXX-XXXX-XXXX-XXXX",
            "instance_id": "inst-001",
            "machine_id":  "completely-different-machine-id",
            "validated_at": "2026-01-01T00:00:00+00:00",
            "expires_grace": "2099-01-01T00:00:00+00:00",
            "customer_email": "x@x.com",
            "product_name": "Orcastrading",
        })
        with patch.dict(os.environ, {"ORCA_DEV_MODE": "0"}):
            ok, msg = lic.check()
        assert ok is False
        assert msg == "wrong_machine"

    def test_offline_grace_accepted(self, tmp_path, monkeypatch):
        from core import license as lic
        from datetime import datetime, timedelta, timezone

        monkeypatch.setattr(lic, "_CACHE_DIR",  tmp_path)
        monkeypatch.setattr(lic, "_CACHE_FILE", tmp_path / "license.json")
        monkeypatch.setattr(lic, "get_machine_id", lambda: "test-machine-id")

        future_grace = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        lic._save_cache({
            "license_key": "YYYY-YYYY-YYYY-YYYY",
            "instance_id": "inst-002",
            "machine_id":  "test-machine-id",
            "validated_at": "2026-01-01T00:00:00+00:00",
            "expires_grace": future_grace,
            "customer_email": "test@test.com",
            "product_name": "Orcastrading",
        })

        # Make the online validation fail (simulate offline)
        with patch.dict(os.environ, {"ORCA_DEV_MODE": "0"}):
            with patch.object(lic, "_ls_validate", return_value=(False, "offline:connection refused")):
                ok, msg = lic.check()

        assert ok is True
        assert "offline" in msg


class TestGetInfo:
    def test_returns_none_without_cache(self, tmp_path, monkeypatch):
        from core import license as lic
        monkeypatch.setattr(lic, "_CACHE_FILE", tmp_path / "no.json")
        assert lic.get_info() is None

    def test_returns_dict_with_cache(self, tmp_path, monkeypatch):
        from core import license as lic
        monkeypatch.setattr(lic, "_CACHE_DIR",  tmp_path)
        monkeypatch.setattr(lic, "_CACHE_FILE", tmp_path / "license.json")
        lic._save_cache({
            "license_key": "ABCD-EFGH-IJKL-MNOP",
            "instance_id": "inst-x",
            "machine_id":  "abc",
            "validated_at": "2026-01-01T00:00:00",
            "expires_grace": "2026-01-08T00:00:00",
            "customer_email": "trader@example.com",
            "product_name": "Orcastrading",
        })
        info = lic.get_info()
        assert info is not None
        assert info["email"] == "trader@example.com"
        assert info["key_preview"].endswith("MNOP")
        assert "machine_id" in info
