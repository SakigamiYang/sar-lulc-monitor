"""Network utilities for environments with broken IPv6 connectivity."""

import socket

_original_getaddrinfo = socket.getaddrinfo
_ipv4_enabled = False


def enable_ipv4_only() -> None:
    """Force IPv4 for Python DNS lookups. Safe to call repeatedly."""
    global _ipv4_enabled

    if _ipv4_enabled:
        return

    def ipv4_getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=0,
            proto=0,
            flags=0,
    ):
        if family == socket.AF_UNSPEC:
            family = socket.AF_INET

        return _original_getaddrinfo(
            host, port, family, type, proto, flags
        )

    socket.getaddrinfo = ipv4_getaddrinfo
    _ipv4_enabled = True


def disable_ipv4_only() -> None:
    """Restore the original DNS lookup behavior."""
    global _ipv4_enabled

    if not _ipv4_enabled:
        return

    socket.getaddrinfo = _original_getaddrinfo
    _ipv4_enabled = False
