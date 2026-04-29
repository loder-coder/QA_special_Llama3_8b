from __future__ import annotations

import os
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import Header, HTTPException, Request


logger = logging.getLogger(__name__)
_unauthenticated_local_warning_logged = False


@dataclass
class InMemoryRateLimiter:
    max_requests: int = 60
    window_seconds: int = 60
    requests: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self.requests[key]
        while bucket and now - bucket[0] >= self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True


def configured_api_key() -> str:
    return os.getenv("ALTONG_API_KEY", "").strip()


def allow_unauthenticated_local() -> bool:
    requested = os.getenv("ALTONG_ALLOW_UNAUTHENTICATED_LOCAL", "").strip().lower() == "true"
    if not requested:
        return False
    environment = os.getenv("ALTONG_ENV", os.getenv("ENVIRONMENT", "")).strip().lower()
    if environment in {"prod", "production"}:
        logger.error("ALTONG_ALLOW_UNAUTHENTICATED_LOCAL was requested but is disabled in production")
        return False
    global _unauthenticated_local_warning_logged
    if not _unauthenticated_local_warning_logged:
        logger.warning("ALTONG_ALLOW_UNAUTHENTICATED_LOCAL=true is enabled; local requests may bypass API key checks")
        _unauthenticated_local_warning_logged = True
    return True


def configured_rate_limit() -> int:
    value = os.getenv("ALTONG_RATE_LIMIT_PER_MINUTE", "60").strip()
    try:
        return max(1, int(value))
    except ValueError:
        return 60


rate_limiter = InMemoryRateLimiter(max_requests=configured_rate_limit())


def validate_auth_configuration() -> None:
    allow_unauthenticated_local()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected_api_key = configured_api_key()
    if not expected_api_key:
        if allow_unauthenticated_local():
            return
        raise HTTPException(status_code=503, detail="api key is not configured")
    if x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="invalid api key")


def require_rate_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_host):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
