import os
import sys
import importlib
import pytest
from fastapi import HTTPException


def _setup_env():
    os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
    os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test-supabase-key")
    os.environ.setdefault("ADMIN_JWT_SECRET", "test-admin-secret")


def test_super_admin_guard_blocks(monkeypatch):
    _setup_env()
    root = os.path.dirname(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    admin_module = importlib.import_module("api.admin")

    # Force guard to think target is super admin
    monkeypatch.setattr(admin_module, "is_super_admin_client", lambda cid: True)
    with pytest.raises(HTTPException) as excinfo:
        admin_module._ensure_not_super_admin("any-id")
    assert excinfo.value.status_code == 403


def test_super_admin_guard_allows_non_super(monkeypatch):
    _setup_env()
    root = os.path.dirname(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    admin_module = importlib.import_module("api.admin")
    monkeypatch.setattr(admin_module, "is_super_admin_client", lambda cid: False)
    # Should not raise
    admin_module._ensure_not_super_admin("some-id")

