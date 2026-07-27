"""Organizer-only request authentication for the evaluator service."""

from __future__ import annotations

import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse


class OrganizerTokenMiddleware(BaseHTTPMiddleware):
    """Protect evaluator POST requests in organizer competition runs."""

    async def dispatch(self, request: Request, call_next):
        expected = os.environ.get("CAR_BENCH_ORGANIZER_TOKEN")
        if expected and request.method == "POST":
            supplied = request.headers.get("X-CAR-BENCH-ORGANIZER-TOKEN", "")
            if not hmac.compare_digest(supplied, expected):
                return PlainTextResponse("Forbidden", status_code=403)
        return await call_next(request)
