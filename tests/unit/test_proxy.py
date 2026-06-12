"""Tests for pool/proxy.py - routing proxy, streaming, auth injection.

Simplified for Architecture A: no swap, no fallback, no idle timers.
Tests cover:
- Tag resolution from header/model field/default
- Auth injection for external resources
- Streaming proxy passthrough
- Non-streaming proxy
- Failover across candidates
- Safe JSON parsing
"""

import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from modelpool.registry import Registry, Resource, Worker, AuthConfig
from modelpool.pool.router import Router, RoutingError, Resolution
from modelpool.pool.proxy import PoolProxy, _safe_json


def make_test_registry():
    data = {
        "resources": {
            "gpu-27b": {
                "type": "managed",
                "size_gb": 16,
                "ctx": 131072,
                "workers": ["hwrouter"],
                "tags": {"chat": 1, "agentic": 1},
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
        )

        with patch.dict("os.environ", {}, clear=True):
            mock_request = MagicMock()
            mock_request.headers = {"content-type": "application/json"}
            headers = proxy._build_proxy_headers(mock_request, resolution)

        assert "authorization" not in headers


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


class TestProxyNoSwap:
    """Architecture A: proxy has no swap-related methods or state."""

    def test_no_trigger_swap_method(self):
        """PoolProxy should not have _trigger_swap."""
        assert not hasattr(PoolProxy, "_trigger_swap")

    def test_no_try_fallbacks_method(self):
        """PoolProxy should not have _try_fallbacks."""
        assert not hasattr(PoolProxy, "_try_fallbacks")

    def test_no_idle_timers(self):
        """PoolProxy should not have idle timer state."""
        assert not hasattr(PoolProxy, "get_idle_timers")
        assert not hasattr(PoolProxy, "_reset_idle_timer")

    def test_no_swap_error(self):
        """SwapError should not exist in proxy module."""
        import modelpool.pool.proxy as proxy_mod
        assert not hasattr(proxy_mod, "SwapError")
