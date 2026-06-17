"""
core/auth.py — Supabase Auth helpers for the Streamlit UI.

Session state keys written on successful login:
  st.session_state["user"]            — {id, email}
  st.session_state["access_token"]    — JWT for API calls
  st.session_state["refresh_token"]   — for token refresh
  st.session_state["supabase_client"] — user-scoped Client (RLS active)

Dev mode (ORCA_DEV_MODE=1 + no SUPABASE_URL):
  Auth gate is bypassed entirely.  A LocalClient backed by the local SQLite
  journal is injected into session state so all UI calls work offline.
"""
from __future__ import annotations
import os
import streamlit as st
from core.supabase_client import get_anon_client, get_user_client


# ── Dev mode helpers ──────────────────────────────────────────────────────────

def _is_dev_mode() -> bool:
    """True when running locally without a Supabase project configured."""
    from core.supabase_client import is_configured
    return (
        os.getenv("ORCA_DEV_MODE", "").strip() in ("1", "true", "yes")
        and not is_configured()
    )


def _inject_dev_session() -> None:
    """Auto-login as a local dev user (called once per Streamlit session)."""
    if st.session_state.get("user"):
        return   # already injected
    from core.local_client import LocalClient
    st.session_state["user"]            = {"id": "dev", "email": "local@dev"}
    st.session_state["access_token"]    = "dev-local"
    st.session_state["refresh_token"]   = "dev-local"
    st.session_state["supabase_client"] = LocalClient()


# ── Session helpers ───────────────────────────────────────────────────────────

def is_authenticated() -> bool:
    if _is_dev_mode():
        _inject_dev_session()
        return True
    return bool(st.session_state.get("user") and st.session_state.get("access_token"))


def get_session_client():
    """Return the user-scoped Supabase client stored in session state."""
    return st.session_state.get("supabase_client")


def _store_session(auth_response) -> None:
    """Write auth response data into Streamlit session state."""
    session = auth_response.session
    user    = auth_response.user
    st.session_state["user"]            = {"id": str(user.id), "email": user.email}
    st.session_state["access_token"]    = session.access_token
    st.session_state["refresh_token"]   = session.refresh_token
    st.session_state["supabase_client"] = get_user_client(
        session.access_token, session.refresh_token
    )


# ── Auth operations ───────────────────────────────────────────────────────────

def login(email: str, password: str) -> str | None:
    """
    Sign in with email + password.
    Returns None on success, or an error message string on failure.
    Writes session data to st.session_state on success.
    """
    try:
        resp = get_anon_client().auth.sign_in_with_password(
            {"email": email.strip(), "password": password}
        )
        _store_session(resp)
        return None
    except Exception as e:
        msg = str(e)
        if "Invalid login credentials" in msg:
            return "Incorrect email or password."
        if "Email not confirmed" in msg:
            return "Please verify your email before logging in."
        return f"Login failed: {msg}"


def register(email: str, password: str) -> str | None:
    """
    Create a new account.
    Returns None on success (email verification sent), or an error string.
    If Supabase is configured with auto-confirm, also logs the user in.
    """
    try:
        resp = get_anon_client().auth.sign_up(
            {"email": email.strip(), "password": password}
        )
        # Auto-confirm: session is present immediately
        if resp.session:
            _store_session(resp)
            return None
        # Email confirmation required
        return "CHECK_EMAIL"
    except Exception as e:
        msg = str(e)
        if "already registered" in msg.lower() or "already been registered" in msg.lower():
            return "An account with this email already exists. Please log in."
        if "Password should be" in msg:
            return "Password must be at least 6 characters."
        return f"Registration failed: {msg}"


def logout() -> None:
    """Sign out and clear session state."""
    if not _is_dev_mode():
        try:
            client = get_session_client()
            if client:
                client.auth.sign_out()
        except Exception:
            pass
    for key in ("user", "access_token", "refresh_token", "supabase_client"):
        st.session_state.pop(key, None)
    st.cache_data.clear()


def reset_password(email: str) -> str | None:
    """
    Send a password reset email.
    Returns None on success, or an error string.
    """
    try:
        get_anon_client().auth.reset_password_email(email.strip())
        return None
    except Exception as e:
        return f"Reset failed: {e}"
