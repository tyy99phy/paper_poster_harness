from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from poster_harness.account_auth import find_account_auth_file, load_account_auth


def _fake_jwt(exp: int) -> str:
    def enc(obj):
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{enc({'alg': 'none'})}.{enc({'exp': exp})}.sig"


def test_auth_discovery_prefers_pro(tmp_path: Path):
    exp = int(time.time()) + 3600
    plus = tmp_path / "codex-user@example.com-plus.json"
    pro = tmp_path / "codex-user@example.com-pro.json"
    plus.write_text(json.dumps({"access_token": _fake_jwt(exp), "account_id": "acct_plus", "email": "user@example.com"}))
    pro.write_text(json.dumps({"access_token": _fake_jwt(exp), "account_id": "acct_pro", "email": "user@example.com"}))

    assert find_account_auth_file(account="user@example.com", auth_dir=tmp_path) == pro
    auth = load_account_auth(account="user@example.com", auth_dir=tmp_path)
    assert auth.account_id == "acct_pro"
    assert auth.email == "user@example.com"
