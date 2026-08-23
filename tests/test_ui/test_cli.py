"""Tests for the ``pyanalytica`` command-line entry point.

These cover argument parsing and port selection only -- the parts that decide
what a student sees when the command misbehaves. Actually serving the app is
covered by the end-to-end suite.
"""

from __future__ import annotations

import socket

import pytest

from pyanalytica.ui.app import _parse_args, _port_is_free, _resolve_port


@pytest.fixture
def busy_port():
    """Bind a port for the duration of a test and yield its number."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Set SO_REUSEADDR to match what uvicorn does. Without it this fixture
    # does not reproduce the real conflict on Windows, where that flag lets a
    # second socket bind the same port.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    yield sock.getsockname()[1]
    sock.close()


class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])
        assert args.port is None      # None means "pick for me"
        assert args.host == "127.0.0.1"
        assert args.no_browser is False

    def test_port_is_parsed_as_int(self):
        assert _parse_args(["--port", "8123"]).port == 8123

    def test_host_override(self):
        assert _parse_args(["--host", "0.0.0.0"]).host == "0.0.0.0"

    def test_no_browser_flag(self):
        assert _parse_args(["--no-browser"]).no_browser is True

    def test_unknown_option_exits(self):
        """Previously every option was silently ignored, including typos."""
        with pytest.raises(SystemExit):
            _parse_args(["--prot", "8000"])

    def test_version_exits_zero(self):
        with pytest.raises(SystemExit) as exc:
            _parse_args(["--version"])
        assert exc.value.code == 0


class TestPortIsFree:
    def test_bound_port_is_not_free(self, busy_port):
        assert _port_is_free("127.0.0.1", busy_port) is False

    def test_unbound_port_is_free(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        # Socket closed, so the port is available again.
        assert _port_is_free("127.0.0.1", port) is True


class TestResolvePort:
    def test_explicit_free_port_is_honoured(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        assert _resolve_port("127.0.0.1", port) == port

    def test_explicit_busy_port_fails_loudly(self, busy_port):
        """Moving someone off the port they asked for would be worse."""
        with pytest.raises(SystemExit) as exc:
            _resolve_port("127.0.0.1", busy_port)
        assert "already in use" in str(exc.value)

    def test_default_is_used_when_free(self, busy_port):
        # busy_port is some arbitrary high port; use it as the "default" here
        # so the test does not depend on whether 8000 is free on this machine.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free = s.getsockname()[1]
        assert _resolve_port("127.0.0.1", None, default=free) == free

    def test_busy_default_falls_back_to_another_port(self, busy_port):
        """The 'address already in use' crash students used to hit."""
        chosen = _resolve_port("127.0.0.1", None, default=busy_port)
        assert chosen != busy_port
        assert _port_is_free("127.0.0.1", chosen)
