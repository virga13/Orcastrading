"""
core/supabase_client.py — Supabase client factory.

Two client modes:
  get_anon_client()        — uses the public anon key; for auth ops (login/register)
  get_user_client(tokens)  — sets the user session so RLS filters by auth.uid()

Reads SUPABASE_URL and SUPABASE_ANON_KEY from .env.
"""
from __future__ import annotations
import os
from functools import lru_cache

from supabase import create_client, Client


def _url() -> str:
    v = os.getenv("SUPABASE_URL", "")
    if not v:
        raise RuntimeError("SUPABASE_URL not set — add it to .env")
    return v


def _anon_key() -> str:
    v = os.getenv("SUPABASE_ANON_KEY", "")
    if not v:
        raise RuntimeError("SUPABASE_ANON_KEY not set — add it to .env")
    return v


@lru_cache(maxsize=1)
def get_anon_client() -> Client:
    """Shared anonymous client for auth operations (login / register)."""
    return create_client(_url(), _anon_key())


def get_user_client(access_token: str, refresh_token: str) -> Client:
    """
    Return a Supabase client authenticated as a specific user.
    All subsequent queries go through RLS, filtered by auth.uid().
    Call this once per Streamlit request using tokens from st.session_state.
    """
    client = create_client(_url(), _anon_key())
    client.auth.set_session(access_token, refresh_token)
    return client


def is_configured() -> bool:
    """Return True if Supabase env vars are present (regardless of validity)."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY"))
