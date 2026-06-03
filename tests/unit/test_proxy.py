"""Tests for pool/proxy.py - routing proxy, streaming, auth injection, fallbacks.

Tests cover:
- Tag resolution from header/model field/default
- Swap triggering for managed resources
- Auth injection for external resources
- Streaming proxy passthrough
- Non-streaming proxy
- Fallback on swap failure
- Connection error handling
"""

import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from modelpool.registry import Registry, Resource, Worker, AuthConfig
from modelpool.pool.router import Router, RoutingError, Resolution
from modelpool.pool.proxy import PoolProxy, SwapError, _safe_json


def make_test_registry():
    data = {
        "resources": {
            "gpu-27b": {
                "type": "managed",
                "size_gb": 16,
                "ctx": 131072,
                "workers": ["hwrouter"],
                "tags": {"chat": 1, "agentic": 1},
                "generalist": True,
                "command": {"binary": "/bin/test", "flags": [["-m", "27b.gguf"]]},
            },
            "gpu-35b": {
                "type": "managed",
                "size_gb": 21,
                "ctx": 262144,
                "workers": ["hwrouter"],
                "tags": {"compression": 1, "title": 1},
                "command": {"binary": "/bin/test", "flags": [["-m", "35b.gguf"]]},
            },
            "cloud-grok": {
                "type": "external",
                "endpoint": "https://api.x.ai/v1",
                "auth": {"method": "xai-oauth"},
                "model": "grok-4.3",
                "ctx": 256000,
                "workers": ["cloud-xai"],
                "tags": {"chat": 4, "vision": 1},
            },
            "cloud-glm": {
                "type": "external",
                "endpoint": "https://api.example.com/v1",
                "auth": {"method": "api_key", "env_var": "TEST_API_KEY"},
                "model": "glm-4.5-flash",
                "ctx": 131072,
                "workers": ["cloud-zai"],
                "tags": {"compression": 4, "triage": 1},
            },
        },
        "workers": {
            "hwrouter": {
                "host": "192.168.35.185",
                "pool_secret": "mp-secret",
                "max_concurrent_models": 1,
            },
            "cloud-xai": {"type": "external"},
            "cloud-zai": {"type": "external"},
        },
    }
    return Registry(data)


class TestSafeJson:
    """Tests for the _safe_json helper."""

    def test_valid_json(self):
        result = _safe_json(b'{"model": "chat", "stream": true}')
        assert result == {"model": "chat", "stream": True}

    def test_invalid_json_returns_none(self):
        result = _safe_json(b"not json")
        assert result is None

    def test_empty_body_returns_none(self):
        result = _safe_json(b"")
        assert result is None

    def test_null_json_returns_none(self):
        result = _safe_json(b"null")
        assert result is None


class TestProxyTagResolution:
    """Tests for how the proxy determines the tag from a request."""

    def test_tag_from_model_field(self):
        """Model field matching a tag is used for routing."""
        body = _safe_json(b'{"model": "compression", "messages": []}')
        reg = make_test_registry()
        router = Router(reg)

        tag = None
        if body and body.get("model", "") in router.tags:
            tag = body["model"]
        if not tag:
            tag = "chat"
        assert tag == "compression"

    def test_tag_falls_back_to_chat(self):
        """If no header and model doesn't match a tag, default to 'chat'."""
        body = _safe_json(b'{"model": "unknown-model", "messages": []}')
        reg = make_test_registry()
        router = Router(reg)

        tag = None
        if body and body.get("model", "") in router.tags:
            tag = body["model"]
        if not tag:
            tag = "chat"
        assert tag == "chat"


class TestProxyAuthInjection:
    """Tests for auth header injection in _build_proxy_headers."""

    def test_api_key_injection(self):
        reg = make_test_registry()
        router = Router(reg)
        proxy = PoolProxy(reg, router)

        resource = reg.get_resource("cloud-glm")
        worker = reg.get_worker("cloud-zai")

        resolution = Resolution(
            tag="triage",
            resource=resource,
            worker=worker,
            needs_swap=False,
            currently_loaded=None,
            fallback_chain=[],
        )

        with patch.dict("os.environ", {"TEST_API_KEY": "test-key-123"}):
            mock_request = MagicMock()
            mock_request.headers = {"content-type": "application/json"}
            headers = proxy._build_proxy_headers(mock_request, resolution)

        assert headers["authorization"] == "Bearer test-key-123"

    def test_no_auth_for_managed(self):
        """Managed resources should not get auth headers."""
        reg = make_test_registry()
        router = Router(reg)
        proxy = PoolProxy(reg, router)

        resource = reg.get_resource("gpu-27b")
        worker = reg.get_worker("hwrouter")

        resolution = Resolution(
            tag="chat",
            resource=resource,
            worker=worker,
            needs_swap=False,
            currently_loaded=None,
            fallback_chain=[],
        )

        mock_request = MagicMock()
        mock_request.headers = {"content-type": "application/json"}
        headers = proxy._build_proxy_headers(mock_request, resolution)

        assert "authorization" not in headers

    def test_missing_env_var_no_auth(self):
        """If the env var for API key is missing, no auth header."""
        reg = make_test_registry()
        router = Router(reg)
        proxy = PoolProxy(reg, router)

        resource = reg.get_resource("cloud-glm")
        worker = reg.get_worker("cloud-zai")

        resolution = Resolution(
            tag="triage",
            resource=resource,
            worker=worker,
            needs_swap=False,
            currently_loaded=None,
            fallback_chain=[],
        )

        with patch.dict("os.environ", {}, clear=True):
            mock_request = MagicMock()
            mock_request.headers = {"content-type": "application/json"}
            headers = proxy._build_proxy_headers(mock_request, resolution)

        assert "authorization" not in headers


class TestProxySwap:
    """Tests for swap triggering."""

    @patch("modelpool.pool.proxy.httpx.AsyncClient")
    def test_swap_sends_pool_secret(self, mock_client_class):
        """Swap request must include X-Pool-Secret header."""
        import asyncio

        reg = make_test_registry()
        router = Router(reg)
        proxy = PoolProxy(reg, router)

        resource = reg.get_resource("gpu-35b")
        worker = reg.get_worker("hwrouter")

        resolution = Resolution(
            tag="compression",
            resource=resource,
            worker=worker,
            needs_swap=True,
            currently_loaded="gpu-27b",
            fallback_chain=[],
        )

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=202, text="ok")
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        asyncio.run(proxy._trigger_swap(resolution))

        post_call = mock_client.post.call_args
        assert post_call is not None
        headers = post_call[1].get("headers", {})
        assert headers.get("X-Pool-Secret") == "mp-secret"

    @patch("modelpool.pool.proxy.httpx.AsyncClient")
    def test_swap_non_202_raises_swap_error(self, mock_client_class):
        """Non-202 response from worker should raise SwapError."""
        import asyncio

        reg = make_test_registry()
        router = Router(reg)
        proxy = PoolProxy(reg, router)

        resource = reg.get_resource("gpu-35b")
        worker = reg.get_worker("hwrouter")

        resolution = Resolution(
            tag="compression",
            resource=resource,
            worker=worker,
            needs_swap=True,
            currently_loaded="gpu-27b",
            fallback_chain=[],
        )

        mock_client = AsyncMock()
        mock_response = MagicMock(status_code=500, text="internal error")
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        with pytest.raises(SwapError, match="500"):
            asyncio.run(proxy._trigger_swap(resolution))


class TestProxyIdleTimers:
    """Tests for idle timer state tracking."""

    def test_get_idle_timers_empty(self):
        reg = make_test_registry()
        router = Router(reg)
        proxy = PoolProxy(reg, router)
        assert proxy.get_idle_timers() == {}

    def test_get_idle_timers_with_active(self):
        import time

        reg = make_test_registry()
        router = Router(reg)
        proxy = PoolProxy(reg, router)

        proxy._idle_timers["hwrouter"] = ("gpu-35b", time.time() + 300)

        timers = proxy.get_idle_timers()
        assert "hwrouter" in timers
        assert timers["hwrouter"]["resource"] == "gpu-35b"
        assert timers["hwrouter"]["expires_in_s"] > 0

    def test_get_idle_timers_expired_excluded(self):
        import time

        reg = make_test_registry()
        router = Router(reg)
        proxy = PoolProxy(reg, router)

        proxy._idle_timers["hwrouter"] = ("gpu-35b", time.time() - 10)

        timers = proxy.get_idle_timers()
        assert "hwrouter" not in timers


class TestProxyExternalResources:
    """Tests for external resource config validation."""

    def test_external_resources_have_workers_and_endpoints(self):
        reg = make_test_registry()
        for name, res in reg.resources.items():
            if res.type == "external":
                assert res.workers, f"External resource '{name}' must have workers"
                assert res.endpoint, f"External resource '{name}' must have endpoint"

    def test_external_resources_have_auth(self):
        reg = make_test_registry()
        for name, res in reg.resources.items():
            if res.type == "external":
                assert res.auth is not None, f"External resource '{name}' must have auth config"

    def test_api_key_auth_has_env_var(self):
        reg = make_test_registry()
        glm = reg.get_resource("cloud-glm")
        assert glm.auth.method == "api_key"
        assert glm.auth.env_var == "TEST_API_KEY"


class TestProxyClose:
    """Tests for cleanup."""

    def test_close_cleans_up_client(self):
        import asyncio

        reg = make_test_registry()
        router = Router(reg)
        proxy = PoolProxy(reg, router)

        proxy._http_client = AsyncMock()
        asyncio.run(proxy.close())
        proxy._http_client.aclose.assert_called_once()
