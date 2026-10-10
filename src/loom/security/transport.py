from __future__ import annotations

import ipaddress
import math
import os
import ssl
import time
from datetime import UTC
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx

from loom.config import Config
from loom.contracts import LoomConfigurationError


def retry_after(value: str | None, *, now: float | None = None) -> float | None:
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            instant = parsedate_to_datetime(value)
            if instant.tzinfo is None:
                instant = instant.replace(tzinfo=UTC)
            delay = max(0.0, instant.timestamp() - (time.time() if now is None else now))
        except (ValueError, TypeError, OverflowError):
            return None
    return delay if math.isfinite(delay) and delay >= 0 else None


def loopback(url: str) -> bool:
    host = urlsplit(url).hostname
    try:
        return bool(host and ipaddress.ip_address(host).is_loopback)
    except ValueError:
        return False


def validate_url(url: str, *, ollama: bool = False) -> str:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise LoomConfigurationError("invalid_endpoint") from None
    if (
        not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or (parts.scheme != "https" and not (ollama and parts.scheme == "http" and loopback(url)))
    ):
        raise LoomConfigurationError("insecure_endpoint")
    if port is not None and port <= 0:
        raise LoomConfigurationError("invalid_endpoint")
    return url


def origin(url: str) -> str:
    validate_url(url)
    parts = urlsplit(url)
    assert parts.hostname
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    return f"https://{host}" + (f":{parts.port}" if parts.port not in (None, 443) else "")


def ssl_context(config: Config) -> ssl.SSLContext:
    ca_file: str | None = None
    if config.transport.ca_bundle_env:
        ca_file = os.environ.get(config.transport.ca_bundle_env)
        if not ca_file:
            raise LoomConfigurationError("ca_bundle_unavailable")
    try:
        context = ssl.create_default_context(cafile=ca_file)
    except (OSError, ssl.SSLError):
        raise LoomConfigurationError("invalid_ca_bundle") from None
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    return context


def client(config: Config, *, timeout: float) -> httpx.AsyncClient:
    context = ssl_context(config)
    return httpx.AsyncClient(
        verify=context,
        trust_env=False,
        follow_redirects=False,
        timeout=timeout,
        transport=httpx.AsyncHTTPTransport(verify=context, retries=0, trust_env=False),
    )
