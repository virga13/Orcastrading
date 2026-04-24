"""
core/license.py — License validation for Orcastrading.

Provider: LemonSqueezy (https://lemonsqueezy.com)
  1. Create a product in your LemonSqueezy store
  2. Set LEMONSQUEEZY_STORE_ID in .env (optional — used only for display)
  3. Enable "License keys" on the product
  4. Distribute the app — users enter their key on first launch

Machine binding:
  Each key can activate on N machines (set in LS product settings).
  Activation is tied to a hardware fingerprint — sharing the key with
  a friend will fail unless the product allows multiple activations.

Offline grace:
  After a successful online validation, the result is cached for 7 days.
  If LS is unreachable, the app continues to work until the grace expires.

Dev bypass:
  Set ORCA_DEV_MODE=1 in .env to skip all license checks during development.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict

import requests

# ── Config ────────────────────────────────────────────────────────────────────
_CACHE_DIR          = Path.home() / ".orcastrading"
_CACHE_FILE         = _CACHE_DIR / "license.json"
_OFFLINE_GRACE_DAYS = 7
_PRODUCT_NAME       = "Orcastrading"
_TIMEOUT            = 8          # seconds for API calls

_LS_ACTIVATE_URL   = "https://api.lemonsqueezy.com/v1/licenses/activate"
_LS_VALIDATE_URL   = "https://api.lemonsqueezy.com/v1/licenses/validate"
_LS_DEACTIVATE_URL = "https://api.lemonsqueezy.com/v1/licenses/deactivate"


class _Cache(TypedDict):
    license_key:    str
    instance_id:    str
    machine_id:     str
    validated_at:   str   # ISO UTC
    expires_grace:  str   # ISO UTC — offline deadline
    customer_email: str
    product_name:   str


# ── Machine fingerprint ───────────────────────────────────────────────────────

def get_machine_id() -> str:
    """Return a stable, hashed hardware fingerprint for this machine."""
    try:
        import machineid  # pip install py-machineid
        return machineid.hashed_id(_PRODUCT_NAME)
    except ImportError:
        pass
    # Fallback: hostname + username + platform
    import platform, socket
    raw = f"{socket.gethostname()}:{platform.node()}:{os.getenv('USERNAME', os.getenv('USER', 'user'))}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ── Local cache ───────────────────────────────────────────────────────────────

def _load_cache() -> _Cache | None:
    try:
        with open(_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _save_cache(data: _Cache) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _clear_cache() -> None:
    try:
        _CACHE_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ── LemonSqueezy API calls ────────────────────────────────────────────────────

def _ls_activate(license_key: str) -> tuple[bool, str, dict]:
    """Activate a key on this machine. Returns (success, message, raw_data)."""
    machine_id = get_machine_id()
    try:
        resp = requests.post(
            _LS_ACTIVATE_URL,
            data={
                "license_key":   license_key.strip(),
                "instance_name": f"{_PRODUCT_NAME}-{machine_id[:8]}",
            },
            timeout=_TIMEOUT,
        )
        data = resp.json()
        if resp.status_code == 200 and data.get("activated"):
            return True, "Activated successfully", data
        err = data.get("error") or f"HTTP {resp.status_code}"
        return False, err, data
    except requests.exceptions.ConnectionError:
        return False, "No internet connection. Please connect and try again.", {}
    except requests.exceptions.Timeout:
        return False, "Request timed out. Please try again.", {}
    except Exception as e:
        return False, f"Unexpected error: {e}", {}


def _ls_validate(license_key: str, instance_id: str) -> tuple[bool, str]:
    """Validate an existing activation. Returns (valid, message)."""
    try:
        resp = requests.post(
            _LS_VALIDATE_URL,
            data={"license_key": license_key, "instance_id": instance_id},
            timeout=_TIMEOUT,
        )
        data = resp.json()
        if resp.status_code == 200 and data.get("valid"):
            return True, "Valid"
        return False, data.get("error") or "License invalid or revoked"
    except requests.exceptions.ConnectionError:
        return False, "offline"
    except requests.exceptions.Timeout:
        return False, "offline"
    except Exception as e:
        return False, f"offline:{e}"


def _ls_deactivate(license_key: str, instance_id: str) -> tuple[bool, str]:
    """Remove this machine's activation slot (allows transfer to new machine)."""
    try:
        resp = requests.post(
            _LS_DEACTIVATE_URL,
            data={"license_key": license_key, "instance_id": instance_id},
            timeout=_TIMEOUT,
        )
        data = resp.json()
        if resp.status_code == 200 and data.get("deactivated"):
            return True, "Deactivated"
        return False, data.get("error") or "Deactivation failed"
    except Exception as e:
        return False, f"Error: {e}"


# ── Public API ─────────────────────────────────────────────────────────────────

def activate(license_key: str) -> tuple[bool, str]:
    """
    Activate a new license key on this machine.
    On success, saves activation to local cache and returns (True, message).
    On failure, returns (False, error_message).
    """
    ok, msg, data = _ls_activate(license_key)
    if not ok:
        return False, msg

    now      = datetime.now(timezone.utc)
    instance = data.get("instance", {})
    meta     = data.get("meta", {})

    cache: _Cache = {
        "license_key":    license_key.strip().upper(),
        "instance_id":    str(instance.get("id", "")),
        "machine_id":     get_machine_id(),
        "validated_at":   now.isoformat(),
        "expires_grace":  (now + timedelta(days=_OFFLINE_GRACE_DAYS)).isoformat(),
        "customer_email": meta.get("customer_email", ""),
        "product_name":   meta.get("product_name", _PRODUCT_NAME),
    }
    _save_cache(cache)
    email = cache["customer_email"]
    return True, f"Licensed to {email}" if email else "License activated"


def check() -> tuple[bool, str]:
    """
    Check whether this machine is licensed to run Orcastrading.
    Returns (valid: bool, message: str).

    Validation order:
    1. ORCA_DEV_MODE env var → bypass all checks
    2. No cache  → not licensed
    3. Machine ID mismatch → bound to different machine
    4. Online validation → success refreshes grace period
    5. Online failed but within offline grace → accept
    6. Grace expired → not licensed
    """
    if os.getenv("ORCA_DEV_MODE", "").lower() in ("1", "true", "yes"):
        return True, "dev"

    cache = _load_cache()
    if not cache:
        return False, "no_license"

    if cache.get("machine_id") != get_machine_id():
        return False, "wrong_machine"

    # Try online validation
    ok, msg = _ls_validate(cache["license_key"], cache["instance_id"])
    if ok:
        now = datetime.now(timezone.utc)
        cache["validated_at"]  = now.isoformat()
        cache["expires_grace"] = (now + timedelta(days=_OFFLINE_GRACE_DAYS)).isoformat()
        _save_cache(cache)
        return True, "valid"

    # Offline path
    if "offline" in msg.lower():
        try:
            grace = datetime.fromisoformat(cache["expires_grace"])
            if grace.tzinfo is None:
                grace = grace.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) < grace:
                days_left = (grace - datetime.now(timezone.utc)).days + 1
                return True, f"offline:{days_left}"
        except (KeyError, ValueError):
            pass

    return False, msg


def deactivate() -> tuple[bool, str]:
    """
    Remove this machine's activation. The license slot becomes available
    for activation on a new machine.
    """
    cache = _load_cache()
    if not cache:
        return False, "No active license on this machine"

    ok, msg = _ls_deactivate(cache["license_key"], cache["instance_id"])
    if ok:
        _clear_cache()
        return True, "Deactivated. You can now activate on a new machine."
    return False, msg


def get_info() -> dict | None:
    """
    Return display-safe license info, or None if not activated.
    Does NOT call the API — reads from cache only.
    """
    cache = _load_cache()
    if not cache:
        return None
    return {
        "email":        cache.get("customer_email", "—"),
        "product":      cache.get("product_name", _PRODUCT_NAME),
        "machine_id":   cache.get("machine_id", "")[:8],
        "validated_at": cache.get("validated_at", ""),
        "expires_grace": cache.get("expires_grace", ""),
        "key_preview":  _mask_key(cache.get("license_key", "")),
    }


def _mask_key(key: str) -> str:
    """Show only last 4 chars: ****-****-****-ABCD"""
    parts = key.split("-")
    if len(parts) >= 2:
        return "-".join(["****"] * (len(parts) - 1) + [parts[-1]])
    return key[-4:].rjust(len(key), "*") if len(key) > 4 else "****"
