from __future__ import annotations

import base64
import json
import mimetypes
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .account_auth import AuthBundle, load_account_auth
from .config import cfg_get

CHATGPT_CODEX_RESPONSES_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_IMAGE_INSTRUCTIONS = (
    "You are a visual artist creating a premium scientific conference poster. "
    "Render the user's image request using the image_generation tool. "
    "Create a beautiful, editorial-quality design with strong visual hierarchy and artistic polish. "
    "The highest-priority, non-negotiable requirement: every scientific figure area must be a blank placeholder rectangle "
    "with the requested [FIG NN] label and the requested visible aspect ratio. "
    "If placeholder geometry conflicts with decoration, text, symmetry, or artistry, reconstruct the surrounding layout: "
    "reflow copy into shorter legible tiers, move badges/ornaments, and preserve the information target rather than making the poster sparse. "
    "Never render real or fake scientific data in a placeholder. "
    "Figure-containing cards must use light neutral paper-like surfaces; dark cinematic colors may frame or surround them "
    "but must not fill the chart/plot blocks. Beyond that, use full creative freedom."
)
DEFAULT_IMAGE_EDIT_INSTRUCTIONS = (
    "You are editing an existing scientific poster image or placeholder-layout template. "
    "Preserve the poster layout, composition, colors, card geometry, typography style, placeholders, and all real scientific figures if present. "
    "Do not move, crop, redraw, reinterpret, or replace any scientific figure, plot, table, diagram, axis, curve, legend, or data mark. "
    "Only make the user's requested local visual/text/placeholder-template corrections. "
    "Return the complete edited poster image, not a crop."
)
CODEX_VALID_SIZES = {"1024x1024", "1536x1024", "1024x1536", "auto"}
CODEX_VALID_QUALITIES = {"low", "medium", "high"}
SSE_READ_TIMEOUT_S = 120


class ImageBackendError(RuntimeError):
    pass


@dataclass
class ImageResult:
    path: Path
    revised_prompt: str | None = None
    latency_s: float = 0.0
    model: str = ""
    size: str = ""
    quality: str = ""
    index: int = 0
    image_call_id: str | None = None
    response_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def generate_images_from_config(
    *,
    prompt: str,
    out_dir: str | Path,
    basename: str,
    config: Mapping[str, Any],
    model: str | None = None,
    size: str | None = None,
    quality: str | None = None,
    n: int | None = None,
    account: str | None = None,
) -> list[Path]:
    backend = str(cfg_get(dict(config), "image_generation.backend", "chatgpt_account") or "chatgpt_account")
    image_model = model or cfg_get(dict(config), "image_generation.model", "gpt-5.5")
    image_size = size or cfg_get(dict(config), "image_generation.size", "1024x1536")
    image_quality = quality or cfg_get(dict(config), "image_generation.quality", "high")
    variants = int(n if n is not None else cfg_get(dict(config), "image_generation.variants", 1))

    if backend == "chatgpt_account":
        account_cfg = dict(cfg_get(dict(config), "image_generation.account", {}) or {})
        results = generate_with_chatgpt_account(
            prompt=prompt,
            out_dir=out_dir,
            basename=basename,
            model=str(image_model),
            size=str(image_size),
            quality=str(image_quality),
            n=variants,
            account=account or account_cfg.get("account") or None,
            auth_dir=account_cfg.get("auth_dir") or None,
            auth_file=account_cfg.get("auth_file") or None,
            endpoint=account_cfg.get("endpoint") or CHATGPT_CODEX_RESPONSES_ENDPOINT,
            min_token_seconds=int(account_cfg.get("min_token_seconds") or 60),
            proxy=account_cfg.get("proxy"),
        )
        return [item.path for item in results]
    if backend in {"openai_responses", "openai_compatible", "openai_compatible_responses"}:
        responses_cfg = dict(cfg_get(dict(config), "image_generation.openai_responses", {}) or {})
        endpoint = (
            responses_cfg.get("endpoint")
            or cfg_get(dict(config), "image_generation.endpoint", None)
            or responses_cfg.get("base_url")
            or cfg_get(dict(config), "image_generation.base_url", None)
        )
        if endpoint and str(endpoint).rstrip("/").endswith("/v1"):
            endpoint = str(endpoint).rstrip("/") + "/responses"
        if not endpoint:
            raise ImageBackendError("image_generation.openai_responses.endpoint is required for openai_responses backend")
        results = generate_with_openai_responses(
            prompt=prompt,
            out_dir=out_dir,
            basename=basename,
            model=str(image_model),
            size=str(image_size),
            quality=str(image_quality),
            n=variants,
            endpoint=str(endpoint),
            api_key_env=str(responses_cfg.get("api_key_env") or cfg_get(dict(config), "image_generation.api_key_env", "OPENAI_API_KEY")),
            proxy=responses_cfg.get("proxy") or cfg_get(dict(config), "image_generation.proxy", None),
        )
        return [item.path for item in results]

    raise ImageBackendError(
        f"unsupported image_generation.backend={backend!r}. Supported: chatgpt_account, openai_responses."
    )


def edit_image_from_config(
    *,
    image_path: str | Path,
    prompt: str,
    out_dir: str | Path,
    basename: str,
    config: Mapping[str, Any],
    model: str | None = None,
    size: str | None = None,
    quality: str | None = None,
    account: str | None = None,
) -> Path:
    """Edit an existing poster image via the configured image-generation backend."""
    backend = str(cfg_get(dict(config), "image_generation.backend", "chatgpt_account") or "chatgpt_account")
    image_model = model or cfg_get(dict(config), "image_generation.model", "gpt-5.5")
    image_size = size or cfg_get(dict(config), "autoposter.micro_repair.size", None) or cfg_get(dict(config), "image_generation.size", "auto")
    image_quality = quality or cfg_get(dict(config), "autoposter.micro_repair.quality", None) or cfg_get(dict(config), "image_generation.quality", "high")

    if backend == "chatgpt_account":
        account_cfg = dict(cfg_get(dict(config), "image_generation.account", {}) or {})
        results = generate_with_chatgpt_account(
            prompt=prompt,
            out_dir=out_dir,
            basename=basename,
            model=str(image_model),
            size=str(image_size),
            quality=str(image_quality),
            n=1,
            account=account or account_cfg.get("account") or None,
            auth_dir=account_cfg.get("auth_dir") or None,
            auth_file=account_cfg.get("auth_file") or None,
            endpoint=account_cfg.get("endpoint") or CHATGPT_CODEX_RESPONSES_ENDPOINT,
            min_token_seconds=int(account_cfg.get("min_token_seconds") or 60),
            proxy=account_cfg.get("proxy"),
            reference_images=[image_path],
            instructions=DEFAULT_IMAGE_EDIT_INSTRUCTIONS,
        )
        return results[0].path
    if backend in {"openai_responses", "openai_compatible", "openai_compatible_responses"}:
        responses_cfg = dict(cfg_get(dict(config), "image_generation.openai_responses", {}) or {})
        endpoint = (
            responses_cfg.get("endpoint")
            or cfg_get(dict(config), "image_generation.endpoint", None)
            or responses_cfg.get("base_url")
            or cfg_get(dict(config), "image_generation.base_url", None)
        )
        if endpoint and str(endpoint).rstrip("/").endswith("/v1"):
            endpoint = str(endpoint).rstrip("/") + "/responses"
        if not endpoint:
            raise ImageBackendError("image_generation.openai_responses.endpoint is required for openai_responses backend")
        results = generate_with_openai_responses(
            prompt=prompt,
            out_dir=out_dir,
            basename=basename,
            model=str(image_model),
            size=str(image_size),
            quality=str(image_quality),
            n=1,
            endpoint=str(endpoint),
            api_key_env=str(responses_cfg.get("api_key_env") or cfg_get(dict(config), "image_generation.api_key_env", "OPENAI_API_KEY")),
            proxy=responses_cfg.get("proxy") or cfg_get(dict(config), "image_generation.proxy", None),
            reference_images=[image_path],
            instructions=DEFAULT_IMAGE_EDIT_INSTRUCTIONS,
        )
        return results[0].path
    raise ImageBackendError(
        f"unsupported image_generation.backend={backend!r}. Supported: chatgpt_account, openai_responses."
    )


def generate_with_chatgpt_account(
    *,
    prompt: str,
    out_dir: str | Path,
    basename: str,
    model: str = "gpt-5.5",
    size: str = "1024x1536",
    quality: str = "high",
    n: int = 1,
    account: str | None = None,
    auth_dir: str | Path | None = None,
    auth_file: str | Path | None = None,
    endpoint: str = CHATGPT_CODEX_RESPONSES_ENDPOINT,
    min_token_seconds: int = 60,
    proxy: str | None = None,
    max_retries: int = 1,
    reference_images: Iterable[str | Path] | None = None,
    instructions: str | None = None,
) -> list[ImageResult]:
    auth = load_account_auth(
        account=account,
        auth_dir=auth_dir,
        auth_file=auth_file,
        min_remaining_s=min_token_seconds,
    )
    transport = ChatGPTImageTransport(endpoint=endpoint, proxy=proxy)
    return transport.generate(
        prompt=prompt,
        auth=auth,
        out_dir=Path(out_dir),
        basename=basename,
        model=model,
        size=size,
        quality=quality,
        n=n,
        max_retries=max_retries,
        reference_images=reference_images,
        instructions=instructions or DEFAULT_IMAGE_INSTRUCTIONS,
    )


def generate_with_openai_responses(
    *,
    prompt: str,
    out_dir: str | Path,
    basename: str,
    model: str = "gpt-5.5",
    size: str = "1024x1536",
    quality: str = "high",
    n: int = 1,
    endpoint: str,
    api_key_env: str = "OPENAI_API_KEY",
    proxy: str | None = None,
    max_retries: int = 1,
    reference_images: Iterable[str | Path] | None = None,
    instructions: str | None = None,
) -> list[ImageResult]:
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ImageBackendError(f"{api_key_env} is not set")
    transport = OpenAIResponsesImageTransport(endpoint=endpoint, api_key=api_key, proxy=proxy)
    return transport.generate(
        prompt=prompt,
        out_dir=Path(out_dir),
        basename=basename,
        model=model,
        size=size,
        quality=quality,
        n=n,
        max_retries=max_retries,
        reference_images=reference_images,
        instructions=instructions or DEFAULT_IMAGE_INSTRUCTIONS,
    )



class ChatGPTImageTransport:
    def __init__(self, *, endpoint: str = CHATGPT_CODEX_RESPONSES_ENDPOINT, proxy: str | None = None):
        self.endpoint = endpoint
        self.proxy = proxy

    def generate(
        self,
        *,
        prompt: str,
        auth: AuthBundle,
        out_dir: Path,
        basename: str,
        model: str,
        size: str,
        quality: str,
        n: int,
        max_retries: int = 1,
        reference_images: Iterable[str | Path] | None = None,
        instructions: str = DEFAULT_IMAGE_INSTRUCTIONS,
    ) -> list[ImageResult]:
        if size not in CODEX_VALID_SIZES:
            raise ImageBackendError(
                f"chatgpt_account backend size must be one of {sorted(CODEX_VALID_SIZES)}; got {size!r}."
            )
        if quality not in CODEX_VALID_QUALITIES:
            raise ImageBackendError(f"quality must be one of {sorted(CODEX_VALID_QUALITIES)}; got {quality!r}")
        if not 1 <= int(n) <= 4:
            raise ImageBackendError("variants/n must be between 1 and 4 for chatgpt_account")
        opener = self._opener()
        all_results: list[ImageResult] = []
        for index in range(int(n)):
            per_call_base = basename if int(n) == 1 else f"{basename}-{index + 1}"
            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    body = build_image_request_body(
                        prompt=prompt,
                        model=model,
                        size=size,
                        quality=quality,
                        input_images=reference_images,
                        instructions=instructions,
                    )
                    events = self._post_stream(opener=opener, auth=auth, body=body)
                    started = time.time()
                    results = parse_image_events(
                        events,
                        out_dir=out_dir,
                        basename=per_call_base,
                        model=model,
                        size=size,
                        quality=quality,
                        started_at=started,
                        completed_at=time.time(),
                    )
                    if not results:
                        raise ImageBackendError("chatgpt_account streamed response did not contain an image_generation_call result")
                    for item in results:
                        item.index = index
                    all_results.extend(results)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= max_retries:
                        raise
                    time.sleep(3 + attempt * 3)
            else:  # pragma: no cover
                assert last_error is not None
                raise last_error
        if not all_results:
            raise ImageBackendError("chatgpt_account backend returned no image_generation_call result")
        return all_results

    def _opener(self):
        proxy = self.proxy
        if proxy is None:
            proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""
        handlers = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        return urllib.request.build_opener(*handlers)

    def _post_stream(self, *, opener, auth: AuthBundle, body: Mapping[str, Any]) -> list[dict[str, Any]]:
        req = build_chatgpt_request(endpoint=self.endpoint, auth=auth, body=body, accept_sse=True)
        try:
            response = opener.open(req, timeout=SSE_READ_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            detail = (exc.read() or b"")[:2000].decode("utf-8", "replace")
            raise ImageBackendError(f"chatgpt_account HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ImageBackendError(f"chatgpt_account network error: {exc}") from exc
        except socket.timeout as exc:
            raise ImageBackendError("chatgpt_account connection timed out") from exc
        with response:
            return list(iter_sse_events(response))


class OpenAIResponsesImageTransport:
    def __init__(self, *, endpoint: str, api_key: str, proxy: str | None = None):
        self.endpoint = endpoint
        self.api_key = api_key
        self.proxy = proxy

    def generate(
        self,
        *,
        prompt: str,
        out_dir: Path,
        basename: str,
        model: str,
        size: str,
        quality: str,
        n: int,
        max_retries: int = 1,
        reference_images: Iterable[str | Path] | None = None,
        instructions: str = DEFAULT_IMAGE_INSTRUCTIONS,
    ) -> list[ImageResult]:
        if size not in CODEX_VALID_SIZES:
            raise ImageBackendError(
                f"openai_responses backend size must be one of {sorted(CODEX_VALID_SIZES)}; got {size!r}."
            )
        if quality not in CODEX_VALID_QUALITIES:
            raise ImageBackendError(f"quality must be one of {sorted(CODEX_VALID_QUALITIES)}; got {quality!r}")
        if not 1 <= int(n) <= 4:
            raise ImageBackendError("variants/n must be between 1 and 4 for openai_responses")
        opener = self._opener()
        all_results: list[ImageResult] = []
        for index in range(int(n)):
            per_call_base = basename if int(n) == 1 else f"{basename}-{index + 1}"
            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    body = build_image_request_body(
                        prompt=prompt,
                        model=model,
                        size=size,
                        quality=quality,
                        input_images=reference_images,
                        instructions=instructions,
                    )
                    events = self._post_stream(opener=opener, body=body)
                    started = time.time()
                    results = parse_image_events(
                        events,
                        out_dir=out_dir,
                        basename=per_call_base,
                        model=model,
                        size=size,
                        quality=quality,
                        started_at=started,
                        completed_at=time.time(),
                    )
                    if not results:
                        raise ImageBackendError("openai_responses streamed response did not contain an image_generation_call result")
                    for item in results:
                        item.index = index
                    all_results.extend(results)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= max_retries:
                        raise
                    time.sleep(3 + attempt * 3)
            else:  # pragma: no cover
                assert last_error is not None
                raise last_error
        if not all_results:
            raise ImageBackendError("openai_responses backend returned no image_generation_call result")
        return all_results

    def _opener(self):
        proxy = self.proxy
        if proxy is None:
            proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""
        handlers = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        return urllib.request.build_opener(*handlers)

    def _post_stream(self, *, opener, body: Mapping[str, Any]) -> list[dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "poster-harness/0.1",
        }
        req = urllib.request.Request(self.endpoint, method="POST", data=json.dumps(dict(body)).encode("utf-8"), headers=headers)
        try:
            response = opener.open(req, timeout=SSE_READ_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            detail = (exc.read() or b"")[:2000].decode("utf-8", "replace")
            raise ImageBackendError(f"openai_responses HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ImageBackendError(f"openai_responses network error: {exc}") from exc
        except socket.timeout as exc:
            raise ImageBackendError("openai_responses connection timed out") from exc
        with response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type.lower():
                return list(iter_sse_events(response))
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
            return _response_payload_to_done_events(payload)


def build_image_request_body(
    *,
    prompt: str,
    model: str,
    size: str,
    quality: str,
    input_images: Iterable[str | Path] | None = None,
    instructions: str = DEFAULT_IMAGE_INSTRUCTIONS,
) -> dict[str, Any]:
    tool: dict[str, Any] = {"type": "image_generation", "quality": quality}
    if size != "auto":
        tool["size"] = size
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for image_path in input_images or []:
        content.append({"type": "input_image", "image_url": encode_image_as_data_url(image_path), "detail": "high"})
    return {
        "model": model,
        "stream": True,
        "instructions": instructions,
        "input": [{"type": "message", "role": "user", "content": content}],
        "tools": [tool],
        "store": False,
    }


def encode_image_as_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_chatgpt_request(*, endpoint: str, auth: AuthBundle, body: Mapping[str, Any], accept_sse: bool) -> urllib.request.Request:
    headers = {
        "Authorization": f"Bearer {auth.access_token}",
        "Chatgpt-Account-Id": auth.account_id,
        "OpenAI-Beta": "responses=v1",
        "Originator": "codex_cli_rs",
        "User-Agent": "codex_cli_rs/1.0.0 poster-harness/0.1",
        "Content-Type": "application/json",
    }
    if accept_sse:
        headers["Accept"] = "text/event-stream"
    return urllib.request.Request(endpoint, method="POST", data=json.dumps(dict(body)).encode("utf-8"), headers=headers)


def iter_sse_events(reader) -> Iterator[dict[str, Any]]:
    buf = b""
    while True:
        try:
            chunk = reader.read(8192)
        except socket.timeout as exc:
            raise ImageBackendError(f"no SSE data from upstream for {SSE_READ_TIMEOUT_S}s") from exc
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            raw, buf = buf.split(b"\n\n", 1)
            for line in raw.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    return
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event


def _response_payload_to_done_events(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if payload.get("id"):
        events.append({"type": "response.completed", "response": dict(payload)})
    for item in payload.get("output", []) if isinstance(payload.get("output"), list) else []:
        if isinstance(item, Mapping):
            events.append({"type": "response.output_item.done", "item": dict(item)})
    return events


def parse_image_events(
    events: Iterable[Mapping[str, Any]],
    *,
    out_dir: Path,
    basename: str,
    model: str,
    size: str,
    quality: str,
    started_at: float,
    completed_at: float,
) -> list[ImageResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ImageResult] = []
    response_id: str | None = None
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type in {"response.created", "response.completed"}:
            response = event.get("response") or {}
            if isinstance(response, Mapping):
                response_id = str(response.get("id") or response_id or "") or None
            continue
        if event_type == "error":
            raise ImageBackendError(str(event.get("message") or event))
        if event_type != "response.output_item.done":
            continue
        item = event.get("item") or {}
        if not isinstance(item, Mapping) or item.get("type") != "image_generation_call":
            continue
        payload = item.get("result")
        if not payload:
            raise ImageBackendError("image_generation_call.done without result")
        filename = basename if not results else f"{basename}-{len(results) + 1}"
        path = _unique_output_path(out_dir, filename, "png")
        path.write_bytes(base64.b64decode(str(payload)))
        results.append(
            ImageResult(
                path=path,
                revised_prompt=item.get("revised_prompt") if isinstance(item.get("revised_prompt"), str) else None,
                latency_s=completed_at - started_at,
                model=model,
                size=size,
                quality=quality,
                index=len(results),
                image_call_id=str(item.get("id") or "") or None,
                response_id=response_id,
            )
        )
    return results


def _unique_output_path(out_dir: Path, basename: str, ext: str) -> Path:
    ext = ext.lower().lstrip(".") or "png"
    path = out_dir / f"{basename}.{ext}"
    idx = 2
    while path.exists():
        path = out_dir / f"{basename}-dup{idx}.{ext}"
        idx += 1
    return path
