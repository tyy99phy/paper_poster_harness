from __future__ import annotations

import base64
import json
import re
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_AUTH_DIR = Path("~/.config/poster-harness/auth")
DEFAULT_MIN_REMAINING_S = 60
_PLAN_PRIORITY = {"pro": 0, "team": 1, "plus": 2, "free": 3}


class AuthError(RuntimeError):
    """Base class for ChatGPT/Codex account auth-file failures."""


class AuthFileMissingError(AuthError):
    pass


class AuthFileCorruptError(AuthError):
    pass


class TokenNearExpiryError(AuthError):
    pass


@dataclass(frozen=True)
class AuthBundle:
    access_token: str
    account_id: str
    email: str
    expires_at: float
    source_path: Path

    def redacted(self) -> dict[str, Any]:
        return {
            "email": self.email,
            "account_id": _redact_middle(self.account_id),
            "expires_at": self.expires_at,
            "source_path": str(self.source_path),
        }


def load_account_auth(
    *,
    account: str | None = None,
    auth_dir: str | Path | None = None,
    auth_file: str | Path | None = None,
    min_remaining_s: int = DEFAULT_MIN_REMAINING_S,
) -> AuthBundle:
    """Load a minimal ChatGPT/Codex account auth JSON file.

    The loader consumes a JSON file with ``access_token`` and ``account_id``
    fields. If a ``refresh_token`` is present and the access token is close to
    expiry, it refreshes the JSON in place. If ``auth_file`` is omitted, files
    matching common account-auth names are discovered under ``auth_dir``. If
    ``account`` is omitted too, the best available local account is selected by
    plan priority (pro, team, plus, free) and then newest mtime.
    """

    account = account or os.getenv("POSTER_HARNESS_ACCOUNT") or None
    auth_file = auth_file or os.getenv("POSTER_HARNESS_AUTH_FILE") or None
    auth_dir = auth_dir or os.getenv("POSTER_HARNESS_AUTH_DIR") or None
    auth_path = Path(auth_file).expanduser() if auth_file else find_account_auth_file(account=account, auth_dir=auth_dir)
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AuthFileMissingError(f"cannot read auth file {auth_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AuthFileCorruptError(f"auth file {auth_path} is not valid JSON: {exc}") from exc

    data = _refresh_if_needed(auth_path, data, min_remaining_s=min_remaining_s)

    access_token = str(data.get("access_token") or "")
    account_id = str(data.get("account_id") or "")
    email = str(data.get("email") or account or infer_email_from_auth_path(auth_path) or "")
    if not access_token or not account_id:
        raise AuthFileCorruptError(f"auth file {auth_path} must contain access_token and account_id")

    expires_at = jwt_exp(access_token)
    remaining = expires_at - time.time()
    if remaining < min_remaining_s:
        raise TokenNearExpiryError(
            f"auth token for {email or auth_path.name} has only {remaining:.0f}s left "
            f"(< {min_remaining_s}s). Refresh the local Codex/ChatGPT login, then retry."
        )
    return AuthBundle(
        access_token=access_token,
        account_id=account_id,
        email=email,
        expires_at=expires_at,
        source_path=auth_path,
    )


def find_account_auth_file(*, account: str | None = None, auth_dir: str | Path | None = None) -> Path:
    root = Path(auth_dir or DEFAULT_AUTH_DIR).expanduser()
    if not root.exists():
        raise AuthFileMissingError(f"auth_dir does not exist: {root}. Set auth_file, auth_dir, POSTER_HARNESS_AUTH_FILE, or POSTER_HARNESS_AUTH_DIR.")
    if account:
        patterns = [f"chatgpt-{account}-*.json", f"chatgpt-{account}.json", f"codex-{account}-*.json", f"codex-{account}.json", f"account-{account}-*.json", f"account-{account}.json", f"{account}.json"]
    else:
        patterns = ["chatgpt-*-*.json", "chatgpt-*.json", "codex-*-*.json", "codex-*.json", "account-*-*.json", "account-*.json", "*.json"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in root.glob(pattern) if path.is_file())
    candidates = _dedupe(candidates)
    if not candidates:
        if account:
            raise AuthFileMissingError(f"no auth file for account {account!r} under {root}")
        raise AuthFileMissingError(f"no auth files found under {root}")
    candidates.sort(key=_auth_file_sort_key)
    return candidates[0]


def list_account_auth_files(auth_dir: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(auth_dir or DEFAULT_AUTH_DIR).expanduser()
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": str(path),
                "email": infer_email_from_auth_path(path),
                "plan": infer_plan_from_auth_path(path),
                "mtime": path.stat().st_mtime,
            }
        )
    rows.sort(key=lambda row: (_PLAN_PRIORITY.get(str(row.get("plan") or ""), 99), -float(row.get("mtime") or 0)))
    return rows


def infer_email_from_auth_path(path: str | Path) -> str:
    stem = Path(path).stem
    match = re.match(r"(?:chatgpt|codex|account)-(.+)-([A-Za-z0-9_]+)$", stem)
    if match:
        return match.group(1)
    match = re.match(r"(?:chatgpt|codex|account)-(.+)$", stem)
    if match:
        return match.group(1)
    if "@" in stem:
        return stem
    return ""


def infer_plan_from_auth_path(path: str | Path) -> str:
    stem = Path(path).stem
    match = re.match(r"(?:chatgpt|codex|account)-.+-([A-Za-z0-9_]+)$", stem)
    return match.group(1).lower() if match else ""


def jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthFileCorruptError(f"access_token is not a JWT (parts={len(parts)})")
    segment = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(segment.encode("ascii")))
    except Exception as exc:
        raise AuthFileCorruptError(f"failed to decode JWT payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuthFileCorruptError("JWT payload is not a JSON object")
    return payload


def jwt_exp(token: str) -> float:
    exp = jwt_payload(token).get("exp")
    if not exp:
        raise AuthFileCorruptError("JWT payload missing exp")
    return float(exp)


def _refresh_if_needed(path: Path, data: dict[str, Any], *, min_remaining_s: int) -> dict[str, Any]:
    try:
        from .auth_login import refresh_auth_json_if_needed

        refreshed = refresh_auth_json_if_needed(path, data, min_remaining_s=min_remaining_s)
        return refreshed if isinstance(refreshed, dict) else data
    except Exception:
        # Strict mode will still fail below if the access token is near expiry.
        return data


def _auth_file_sort_key(path: Path) -> tuple[int, float, str]:
    plan = infer_plan_from_auth_path(path)
    return (_PLAN_PRIORITY.get(plan, 99), -path.stat().st_mtime, str(path))


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        try:
            key = path.resolve()
        except Exception:
            key = path.absolute()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _redact_middle(value: str, *, keep: int = 4) -> str:
    if len(value) <= keep * 2:
        return "***"
    return f"{value[:keep]}...{value[-keep:]}"
