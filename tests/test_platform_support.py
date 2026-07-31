import sysconfig

import pytest

from myclaw.platform_support import (
    SUPPORTED_PLATFORM_TAG,
    UnsupportedPlatformError,
    current_platform_tag,
    normalize_platform_tag,
    require_supported_platform,
)


@pytest.mark.parametrize("value", ("win_amd64", "win-amd64", "WIN-AMD64"))
def test_supported_platform_spellings_normalize_to_windows_x64(value: str) -> None:
    assert normalize_platform_tag(value) == SUPPORTED_PLATFORM_TAG
    require_supported_platform(value)


@pytest.mark.parametrize(
    ("value", "normalized"),
    (
        ("linux-x86_64", "linux_x86_64"),
        ("win32", "win32"),
        ("win-arm64", "win_arm64"),
    ),
)
def test_unsupported_platforms_report_the_detected_process_tag(
    value: str,
    normalized: str,
) -> None:
    with pytest.raises(UnsupportedPlatformError) as captured:
        require_supported_platform(value)

    assert captured.value.platform_tag == normalized
    assert captured.value.error.code == "unsupported_platform"
    assert SUPPORTED_PLATFORM_TAG in captured.value.error.message
    assert normalized in captured.value.error.message


def test_current_platform_tag_is_an_injectable_sysconfig_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sysconfig, "get_platform", lambda: "WIN-AMD64")

    assert current_platform_tag() == SUPPORTED_PLATFORM_TAG
