from __future__ import annotations

import base64
import copy
import json
import mimetypes
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error, request

from .schemas import DEFAULT_MODEL


def normalize_schema_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
    return cleaned or "poster_harness_schema"


def encode_image_as_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def extract_response_text(payload: Mapping[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    chunks: list[str] = []
    for text in _walk_text_nodes(payload):
        text = text.strip()
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def extract_json_from_response(payload: Mapping[str, Any]) -> Any:
    if isinstance(payload.get("output_parsed"), (dict, list)):
        return copy.deepcopy(payload["output_parsed"])
    text = extract_response_text(payload)
    if text:
        return extract_json_from_text(text)
    for output in payload.get("output", []) if isinstance(payload.get("output"), list) else []:
        if isinstance(output, Mapping):
            for content in output.get("content", []) if isinstance(output.get("content"), list) else []:
                if isinstance(content, Mapping):
                    candidate = content.get("json")
                    if isinstance(candidate, (dict, list)):
                        return copy.deepcopy(candidate)
    raise ValueError("No JSON output found in response payload")


def extract_json_from_text(text: str) -> Any:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty text; cannot extract JSON")
    candidates = [cleaned]
    candidates.extend(_json_codeblocks(cleaned))
    candidates.extend(_balanced_json_fragments(cleaned))
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not extract valid JSON from model output")


def _json_codeblocks(text: str) -> list[str]:
    return [match.group(1) for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)]


def _balanced_json_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    for opener in "[{":
        start = text.find(opener)
        while start >= 0:
            fragment = _extract_balanced_fragment(text, start)
            if fragment:
                fragments.append(fragment)
            start = text.find(opener, start + 1)
    return fragments


def _extract_balanced_fragment(text: str, start: int) -> str | None:
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in pairs:
            stack.append(pairs[char])
            continue
        if char in "]}":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start : idx + 1]
    return None


def _walk_text_nodes(value: Any):
    if isinstance(value, Mapping):
        node_type = value.get("type")
        if node_type in {"output_text", "text", "message"} and isinstance(value.get("text"), str):
            yield value["text"]
        elif isinstance(value.get("output_text"), str):
            yield value["output_text"]
        elif isinstance(value.get("content"), str):
            yield value["content"]
        for item in value.values():
            yield from _walk_text_nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_text_nodes(item)

@dataclass(slots=True)
class ChatGPTAccountResponsesProvider:
    """Responses provider backed by a local ChatGPT/Codex account auth JSON.

    This is the minimal account-import path: it only reads an existing auth JSON
    from the configured path or POSTER_HARNESS_AUTH_FILE.
    """

    account: str | None = None
    auth_dir: str | Path | None = None
    auth_file: str | Path | None = None
    model: str = DEFAULT_MODEL
    endpoint: str = "https://chatgpt.com/backend-api/codex/responses"
    timeout: int = 120
    min_token_seconds: int = 60
    proxy: str | None = None
    default_image_detail: str = "high"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "name": "chatgpt_account_responses",
            "model": self.model,
            "configured": True,
            "strict_llm": True,
            "account": self.account or "auto",
        }

    def generate_json(
        self,
        *,
        stage_name: str,
        prompt: str,
        schema: Mapping[str, Any],
        system_prompt: str | None = None,
        image_paths: Sequence[str | Path] | None = None,
        image_detail: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        include: Sequence[str] | None = None,
        reasoning: Mapping[str, Any] | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        schema_copy = copy.deepcopy(dict(schema))
        try:
            body = self._build_request_body(
                stage_name=stage_name,
                prompt=prompt,
                schema=schema_copy,
                system_prompt=system_prompt,
                image_paths=image_paths,
                image_detail=image_detail,
                tools=tools,
                tool_choice=tool_choice,
                include=include,
                reasoning=reasoning,
                strict=strict,
                use_schema=True,
            )
            payload = self._post(body)
            parsed = extract_json_from_response(payload)
            raw_text = extract_response_text(payload)
        except Exception as exc:
            raise RuntimeError(f"{stage_name}: strict ChatGPT account Responses request failed: {exc}") from exc

        if not isinstance(parsed, Mapping):
            raise RuntimeError(f"{stage_name}: strict LLM stage returned non-object JSON: {type(parsed).__name__}")
        return {
            "stage": stage_name,
            "ok": True,
            "mode": "chatgpt_account",
            "provider": self.describe(),
            "prompt": prompt,
            "schema": schema_copy,
            "result": parsed,
            "raw_text": raw_text,
            "response_id": payload.get("id"),
            "warnings": [],
        }

    def _build_request_body(
        self,
        *,
        stage_name: str,
        prompt: str,
        schema: Mapping[str, Any],
        system_prompt: str | None,
        image_paths: Sequence[str | Path] | None,
        image_detail: str | None,
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: str | Mapping[str, Any] | None,
        include: Sequence[str] | None,
        reasoning: Mapping[str, Any] | None,
        strict: bool,
        use_schema: bool,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image_path in image_paths or []:
            content.append(
                {
                    "type": "input_image",
                    "image_url": encode_image_as_data_url(image_path),
                    "detail": image_detail or self.default_image_detail,
                }
            )
        body: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "store": False,
            "input": [{"type": "message", "role": "user", "content": content}],
        }
        if system_prompt:
            body["instructions"] = system_prompt
        if tools:
            body["tools"] = [copy.deepcopy(dict(tool)) for tool in tools]
        if tool_choice:
            body["tool_choice"] = copy.deepcopy(tool_choice)
        if include:
            body["include"] = list(include)
        if reasoning:
            body["reasoning"] = copy.deepcopy(dict(reasoning))
        if use_schema:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": normalize_schema_name(stage_name),
                    "schema": copy.deepcopy(dict(schema)),
                    "strict": strict,
                }
            }
        return body

    def _post(self, body: Mapping[str, Any]) -> dict[str, Any]:
        from .account_auth import load_account_auth
        from .image_backend import build_chatgpt_request, iter_sse_events

        auth = load_account_auth(
            account=self.account,
            auth_dir=self.auth_dir,
            auth_file=self.auth_file,
            min_remaining_s=self.min_token_seconds,
        )
        req = build_chatgpt_request(endpoint=self.endpoint, auth=auth, body=body, accept_sse=True)
        handlers = []
        proxy = self.proxy or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or ""
        if proxy:
            handlers.append(request.ProxyHandler({"http": proxy, "https": proxy}))
        opener = request.build_opener(*handlers)
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                events = list(iter_sse_events(resp))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"network error: {exc}") from exc
        return response_payload_from_sse_events(events)


def response_payload_from_sse_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed: dict[str, Any] | None = None
    delta_parts: list[str] = []
    done_texts: list[str] = []
    item_texts: list[str] = []
    output_items: list[Any] = []
    response_id: str | None = None
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type == "error":
            raise RuntimeError(str(event.get("message") or event))
        response = event.get("response")
        if isinstance(response, Mapping):
            response_id = str(response.get("id") or response_id or "") or None
        if event_type == "response.completed" and isinstance(response, Mapping):
            completed = dict(response)
            continue
        if event_type == "response.output_text.delta" and isinstance(event.get("delta"), str):
            delta_parts.append(str(event["delta"]))
        if event_type == "response.output_text.done" and isinstance(event.get("text"), str):
            done_texts.append(str(event["text"]))
        item = event.get("item")
        if isinstance(item, Mapping):
            output_items.append(copy.deepcopy(dict(item)))
            text = _extract_text_from_output_item(item)
            if text:
                item_texts.append(text)
    text = "\n".join(done_texts).strip() or "".join(delta_parts).strip() or "\n".join(item_texts).strip()
    payload: dict[str, Any] = dict(completed or {})
    if response_id and not payload.get("id"):
        payload["id"] = response_id
    if output_items and not payload.get("output"):
        payload["output"] = output_items
    if text:
        payload["output_text"] = text
    return payload

def _extract_text_from_output_item(item: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    content = item.get("content")
    if isinstance(content, list):
        for node in content:
            if isinstance(node, Mapping):
                if isinstance(node.get("text"), str):
                    chunks.append(node["text"])
                elif isinstance(node.get("content"), str):
                    chunks.append(node["content"])
    if isinstance(item.get("text"), str):
        chunks.append(str(item["text"]))
    return "\n".join(part for part in chunks if part).strip()
