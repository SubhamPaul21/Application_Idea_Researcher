"""HTTP security headers for the Streamlit server (Streamlit 1.66 sends none of these by default).

The app has no login, so a malicious website must not be able to show it inside a frame and trick the owner into
clicking Approve, Run or Purge (clickjacking). Used by app/server.py through st.App(middleware=...).
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

SECURITY_HEADERS: list[tuple[bytes, bytes]] = [
    # Only pages from this same server may frame the app (Streamlit's own component iframes still work).
    (b"x-frame-options", b"SAMEORIGIN"),
    (b"content-security-policy", b"frame-ancestors 'self'"),
    # Don't send the local app address to external sites when following evidence links.
    (b"referrer-policy", b"no-referrer"),
    (b"x-content-type-options", b"nosniff"),
]
_NAMES = {name for name, _ in SECURITY_HEADERS}

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class SecurityHeadersMiddleware:
    """ASGI middleware that sets SECURITY_HEADERS on every HTTP response (WebSocket traffic passes through)."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message.get("type") == "http.response.start":
                kept = [(k, v) for k, v in message.get("headers", []) if k.lower() not in _NAMES]
                message = {**message, "headers": kept + SECURITY_HEADERS}
            await send(message)

        await self.app(scope, receive, send_with_headers)
