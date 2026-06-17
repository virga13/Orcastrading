"""
core/demo_limit.py — lifetime AI-call budget for the public demo deployment.

Only active when DEMO_MODE=1 is set (the public Streamlit Cloud deployment).
Local/full usage is never restricted. The site owner can unlock unlimited
calls for their own browser session by entering DEMO_OWNER_KEY in the sidebar.

Usage count is persisted to a small JSON file so it survives Streamlit
session reruns/reconnects — it is NOT guaranteed to survive a container
restart on Streamlit Cloud's free tier, but that only ever gives a few
extra free calls, never an open-ended one.
"""
from __future__ import annotations
import os
import json
import threading
from pathlib import Path

import streamlit as st

DEMO_CALL_LIMIT = 2

_LOCK = threading.Lock()
_USAGE_FILE = Path(__file__).resolve().parent.parent / "data" / "demo_usage.json"


def is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes")


def _owner_key() -> str:
    return os.getenv("DEMO_OWNER_KEY", "")


def is_owner() -> bool:
    return bool(st.session_state.get("is_demo_owner"))


def render_owner_unlock() -> None:
    """Sidebar widget letting the site owner unlock unlimited demo usage."""
    if not is_demo_mode() or is_owner():
        return
    with st.sidebar.expander("Owner unlock"):
        key = st.text_input("Owner key", type="password", key="demo_owner_key_input")
        if key and _owner_key() and key == _owner_key():
            st.session_state["is_demo_owner"] = True
            st.rerun()


def _read_count() -> int:
    try:
        return json.loads(_USAGE_FILE.read_text()).get("count", 0)
    except Exception:
        return 0


def _write_count(n: int) -> None:
    _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USAGE_FILE.write_text(json.dumps({"count": n}))


def calls_remaining() -> int:
    if not is_demo_mode() or is_owner():
        return DEMO_CALL_LIMIT
    return max(0, DEMO_CALL_LIMIT - _read_count())


def try_consume(n: int = 1) -> bool:
    """
    Attempt to book n AI calls against the lifetime demo budget.
    Returns True if allowed, False if it would exceed the budget.
    Always True when not in demo mode or when unlocked as owner.
    """
    if not is_demo_mode() or is_owner():
        return True
    with _LOCK:
        current = _read_count()
        if current + n > DEMO_CALL_LIMIT:
            return False
        _write_count(current + n)
        return True


DEMO_LIMIT_MESSAGE = (
    f"This public demo allows {DEMO_CALL_LIMIT} AI calls total, shared across all visitors — "
    "limit reached. Reach out for a live walkthrough with full functionality."
)
