import importlib
import os
import sys

from datetime import datetime, timedelta, timezone


def _prepare_env():
    """Ensure required env vars exist before importing settings/auth."""
    os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
    os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test-supabase-key")
    os.environ.setdefault("ADMIN_JWT_SECRET", "test-admin-secret")


def test_admin_token_round_trip(monkeypatch):
    _prepare_env()
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    auth = importlib.import_module("utils.auth")
    token = auth.create_admin_token("admin-id", "admin@example.com")
    claims = auth.decode_admin_token(token)

    assert claims["sub"] == "admin-id"
    assert claims["email"] == "admin@example.com"
    assert claims["role"]
    assert datetime.fromtimestamp(claims["exp"], tz=timezone.utc) > datetime.now(timezone.utc)


def test_hash_password(monkeypatch):
    _prepare_env()
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    auth = importlib.import_module("utils.auth")
    assert auth.hash_password("secret") == auth.hash_password("secret")
    assert auth.hash_password("secret") != auth.hash_password("other")

