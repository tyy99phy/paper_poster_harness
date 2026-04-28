from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from poster_harness.auth_login import build_account_auth_json, resolve_output_path, sanitize_filename


def _fake_jwt(payload: dict) -> str:
    def enc(obj):
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{enc({'alg': 'none'})}.{enc(payload)}.sig"


def test_build_account_auth_json_from_tokens(tmp_path: Path):
    exp = int(time.time()) + 3600
    access = _fake_jwt({"exp": exp, "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"}})
    ident = _fake_jwt({"https://api.openai.com/profile": {"email": "user@example.com"}})
    payload = build_account_auth_json({"access_token": access, "refresh_token": "refresh", "id_token": ident})
    assert payload["email"] == "user@example.com"
    assert payload["account_id"] == "acct_123"
    assert payload["refresh_token"] == "refresh"
    out = resolve_output_path(payload, out_dir=tmp_path, out_file=None)
    assert out == tmp_path / "chatgpt-user@example.com.json"


def test_sanitize_filename():
    assert sanitize_filename("u/ser@example.com") == "u_ser@example.com"
