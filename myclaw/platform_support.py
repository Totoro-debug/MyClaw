"""Windows x64 execution gate for every command-line entry path."""

from __future__ import annotations

import sysconfig
from typing import Final

from myclaw.errors import ErrorInfo

SUPPORTED_PLATFORM_TAG: Final = "win_amd64"


class UnsupportedPlatformError(Exception):
    """A safe command-line failure for an unsupported Python platform."""

    def __init__(self, platform_tag: str) -> None:
        self.platform_tag = platform_tag
        self.error = ErrorInfo(
            "unsupported_platform",
            "MyClaw requires a 64-bit x86-64 Windows Python process "
            f"({SUPPORTED_PLATFORM_TAG}); detected {platform_tag}.",
        )
        super().__init__(self.error.message)


def normalize_platform_tag(value: str) -> str:
    """Normalize the packaging platform spelling used by Python runtimes."""
    return value.strip().lower().replace("-", "_")


def current_platform_tag() -> str:
    """Return the normalized platform tag for the running Python process."""
    return normalize_platform_tag(sysconfig.get_platform())


def require_supported_platform(platform_tag: str | None = None) -> None:
    """Reject every platform except 64-bit Windows on x86-64."""
    detected = (
        current_platform_tag() if platform_tag is None else normalize_platform_tag(platform_tag)
    )
    if detected != SUPPORTED_PLATFORM_TAG:
        raise UnsupportedPlatformError(detected)
