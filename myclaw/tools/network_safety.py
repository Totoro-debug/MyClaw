"""Shared network target safety assessment for Core Tools."""

from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Literal, Protocol

from myclaw.tools.base import is_public_ip

type TargetRisk = Literal[
    "literal_non_global",
    "dns_failure",
    "dns_empty",
    "dns_non_global",
]


class DNSResolver(Protocol):
    """Resolve every TCP address for one network target."""

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]: ...


class SocketDNSResolver:
    """Resolve every TCP address through the host event loop."""

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        records = await asyncio.get_running_loop().getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        return tuple(dict.fromkeys(record[4][0] for record in records))


@dataclass(frozen=True, slots=True)
class TargetAssessment:
    """The shared safety assessment for one network target."""

    risk: TargetRisk | None = None


async def assess_target(
    hostname: str,
    port: int,
    resolver: DNSResolver,
) -> TargetAssessment:
    """Assess whether one literal or DNS name resolves only to global addresses."""
    try:
        ip_address(hostname)
    except ValueError:
        try:
            answers = await resolver.resolve(hostname, port)
        except Exception:
            return TargetAssessment(risk="dns_failure")
        if answers and all(is_public_ip(answer) for answer in answers):
            return TargetAssessment()
        if answers:
            return TargetAssessment(risk="dns_non_global")
        return TargetAssessment(risk="dns_empty")
    return TargetAssessment(risk=None if is_public_ip(hostname) else "literal_non_global")


__all__ = [
    "DNSResolver",
    "SocketDNSResolver",
    "TargetAssessment",
    "TargetRisk",
    "assess_target",
]
