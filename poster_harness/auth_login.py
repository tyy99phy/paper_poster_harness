from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .account_auth import DEFAULT_AUTH_DIR, jwt_payload

AUTH_ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_ISSUER}/oauth/authorize"
TOKEN_URL = f"{AUTH_ISSUER}/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_CALLBACK_PORT = 1455
DEFAULT_SCOPE = "openid profile email offline_access"


class LoginError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoginResult:
    path: Path
    email: str
    account_id: str
    expires_at: float

    def public_summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "email": self.email,
            "account_id": _redact_middle(self.account_id),
            "expires_at": self.expires_at,
        }


@dataclass
class CallbackResult:
    code: str = ""
    state: str = ""
    error: str = ""
    error_description: str = ""


def run_browser_login(
    *,
    out_dir: str | Path | None = None,
    out_file: str | Path | None = None,
    callback_port: int = DEFAULT_CALLBACK_PORT,
    open_browser: bool = True,
    timeout_s: int = 900,
    force: bool = False,
) -> LoginResult:
    """Interactive browser login that writes a local account-auth JSON.

    The user completes the OpenAI/ChatGPT login in their browser. This function
    only receives the localhost OAuth callback, exchanges the code, and stores
    the resulting tokens in the harness auth JSON format.
    """

    verifier = secrets.token_urlsafe(64)
    challenge = _pkce_challenge(verifier)
    state = secrets.token_urlsafe(32)
    redirect_uri = f"http://localhost:{callback_port}/auth/callback"
    auth_url = build_authorize_url(
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=challenge,
    )

    server = _CallbackServer(("127.0.0.1", callback_port), _CallbackHandler)
    server.expected_state = state

    print("Starting local OpenAI/ChatGPT login.")
    print(f"Listening for OAuth callback on {redirect_uri}")
    if open_browser:
        opened = webbrowser.open(auth_url)
        if not opened:
            print("Could not open a browser automatically. Open this URL manually:")
            print(auth_url)
    else:
        print("Open this URL in your browser:")
        print(auth_url)

    deadline = time.time() + timeout_s
    while not server.result and time.time() < deadline:
        server.handle_request()
    if not server.result:
        raise LoginError(f"timed out waiting for OAuth callback after {timeout_s}s")
    callback = server.result
    if callback.error:
        detail = f": {callback.error_description}" if callback.error_description else ""
        raise LoginError(f"OAuth authorization failed: {callback.error}{detail}")
    if not callback.code:
        raise LoginError("OAuth callback did not include a code")
    if callback.state != state:
        raise LoginError("OAuth state mismatch")

    token_data = exchange_code_for_tokens(
        code=callback.code,
        verifier=verifier,
        redirect_uri=redirect_uri,
    )
    auth_json = build_account_auth_json(token_data)
    path = resolve_output_path(auth_json, out_dir=out_dir, out_file=out_file)
    if path.exists() and not force:
        raise LoginError(f"auth file already exists: {path} (pass --force to overwrite)")
    write_auth_json(auth_json, path)
    return LoginResult(
        path=path,
        email=str(auth_json.get("email") or ""),
        account_id=str(auth_json.get("account_id") or ""),
        expires_at=float(auth_json.get("expires_at") or 0),
    )


def build_authorize_url(*, redirect_uri: str, state: str, code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": DEFAULT_SCOPE,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "prompt": "login",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def exchange_code_for_tokens(*, code: str, verifier: str, redirect_uri: str, timeout_s: int = 60) -> dict[str, Any]:
    body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_URL,
        method="POST",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LoginError(f"token exchange failed with HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise LoginError(f"token exchange network error: {exc}") from exc
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LoginError("token exchange response was not JSON") from exc
    if not isinstance(data, dict) or not data.get("access_token"):
        raise LoginError(f"token exchange response missing access_token; keys={list(data) if isinstance(data, dict) else type(data).__name__}")
    return data


def refresh_tokens(refresh_token: str, *, timeout_s: int = 60) -> dict[str, Any]:
    body = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
            "scope": DEFAULT_SCOPE,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_URL,
        method="POST",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LoginError(f"token refresh failed with HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise LoginError(f"token refresh network error: {exc}") from exc
    data = json.loads(payload)
    if not isinstance(data, dict) or not data.get("access_token"):
        raise LoginError("token refresh response missing access_token")
    return data


def build_account_auth_json(token_data: dict[str, Any], *, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    access_token = str(token_data.get("access_token") or "")
    refresh_token = str(token_data.get("refresh_token") or (previous or {}).get("refresh_token") or "")
    id_token = str(token_data.get("id_token") or (previous or {}).get("id_token") or "")
    if not access_token:
        raise LoginError("missing access_token")

    access_payload = _safe_jwt_payload(access_token)
    id_payload = _safe_jwt_payload(id_token)
    account_id = _extract_account_id(access_payload) or _extract_account_id(id_payload) or str((previous or {}).get("account_id") or "")
    email = _extract_email(id_payload) or _extract_email(access_payload) or str((previous or {}).get("email") or "")
    if not account_id:
        raise LoginError("could not find ChatGPT account id in OAuth token payload")
    if not email:
        email = "account"
    expires_at = float(access_payload.get("exp") or time.time() + float(token_data.get("expires_in") or 0))
    now = int(time.time())
    return {
        "type": "chatgpt_account",
        "email": email,
        "account_id": account_id,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "id_token": id_token,
        "expires_at": expires_at,
        "created_at": (previous or {}).get("created_at") or _iso_utc(now),
        "updated_at": _iso_utc(now),
    }


def refresh_auth_json_if_needed(path: str | Path, data: dict[str, Any], *, min_remaining_s: int) -> dict[str, Any]:
    access_token = str(data.get("access_token") or "")
    refresh_token = str(data.get("refresh_token") or "")
    if not access_token or not refresh_token:
        return data
    try:
        expires_at = float(jwt_payload(access_token).get("exp") or 0)
    except Exception:
        expires_at = 0
    if expires_at - time.time() >= min_remaining_s:
        return data
    refreshed = refresh_tokens(refresh_token)
    updated = build_account_auth_json(refreshed, previous=data)
    write_auth_json(updated, Path(path))
    return updated


def resolve_output_path(auth_json: dict[str, Any], *, out_dir: str | Path | None, out_file: str | Path | None) -> Path:
    if out_file:
        return Path(out_file).expanduser()
    root = Path(out_dir or os.getenv("POSTER_HARNESS_AUTH_DIR") or DEFAULT_AUTH_DIR).expanduser()
    email = sanitize_filename(str(auth_json.get("email") or "account"))
    return root / f"chatgpt-{email}.json"


def write_auth_json(auth_json: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(auth_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(temp, 0o600)
    except OSError:
        pass
    temp.replace(path)


def sanitize_filename(value: str) -> str:
    value = value.strip() or "account"
    return "".join(ch if ch.isalnum() or ch in {"@", ".", "_", "-"} else "_" for ch in value)[:160]


class _CallbackServer(HTTPServer):
    expected_state: str
    result: CallbackResult | None = None
    timeout = 1.0


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in {"/auth/callback", "/callback"}:
            self.send_response(404)
            self.end_headers()
            return
        qs = urllib.parse.parse_qs(parsed.query)
        result = CallbackResult(
            code=(qs.get("code") or [""])[0],
            state=(qs.get("state") or [""])[0],
            error=(qs.get("error") or [""])[0],
            error_description=(qs.get("error_description") or [""])[0],
        )
        self.server.result = result  # type: ignore[attr-defined]
        ok = bool(result.code and result.state == getattr(self.server, "expected_state", ""))
        html = """
        <html><body style="font-family: system-ui; padding: 2rem;">
        <h2>{title}</h2>
        <p>{message}</p>
        <p>You can close this tab and return to the terminal.</p>
        </body></html>
        """.format(
            title="Login complete" if ok else "Login callback received",
            message="The poster harness received the login callback." if ok else "The terminal will validate the callback details.",
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _safe_jwt_payload(token: str) -> dict[str, Any]:
    if not token:
        return {}
    try:
        return jwt_payload(token)
    except Exception:
        return {}


def _extract_account_id(payload: dict[str, Any]) -> str:
    auth = payload.get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        value = auth.get("chatgpt_account_id") or auth.get("account_id")
        if value:
            return str(value)
    for key in ("chatgpt_account_id", "account_id"):
        if payload.get(key):
            return str(payload[key])
    return ""


def _extract_email(payload: dict[str, Any]) -> str:
    profile = payload.get("https://api.openai.com/profile")
    if isinstance(profile, dict) and profile.get("email"):
        return str(profile["email"])
    if payload.get("email"):
        return str(payload["email"])
    return ""


def _iso_utc(ts: int | float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _redact_middle(value: str, *, keep: int = 4) -> str:
    if len(value) <= keep * 2:
        return "***"
    return f"{value[:keep]}...{value[-keep:]}"
