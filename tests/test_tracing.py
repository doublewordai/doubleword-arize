"""Tracing bootstrap tests. phoenix.otel.register is mocked — no Phoenix server required."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _reset_provider():
    """Clear the module-level _provider cache between tests."""
    import dwp.tracing as t
    t._provider = None


def test_local_backend_calls_phoenix_register():
    _reset_provider()
    mock_provider = MagicMock()
    mock_register = MagicMock(return_value=mock_provider)

    with (
        patch("dwp.tracing.settings") as mock_settings,
        patch.dict("sys.modules", {"phoenix.otel": MagicMock(register=mock_register)}),
    ):
        mock_settings.phoenix_collector_endpoint = "http://localhost:6006"
        mock_settings.project_name = "test-project"

        import dwp.tracing as t
        t._provider = None
        provider = t.get_tracer_provider()

    mock_register.assert_called_once_with(
        project_name="test-project",
        auto_instrument=True,
        batch=True,
    )
    assert provider is mock_provider


def test_idempotent_returns_same_provider():
    _reset_provider()
    mock_provider = MagicMock()
    mock_register = MagicMock(return_value=mock_provider)

    with (
        patch("dwp.tracing.settings") as mock_settings,
        patch.dict("sys.modules", {"phoenix.otel": MagicMock(register=mock_register)}),
    ):
        mock_settings.phoenix_collector_endpoint = "http://localhost:6006"
        mock_settings.project_name = "test"

        import dwp.tracing as t
        t._provider = None
        p1 = t.get_tracer_provider()
        p2 = t.get_tracer_provider()

    assert p1 is p2
    assert mock_register.call_count == 1
