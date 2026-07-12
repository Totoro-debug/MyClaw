from importlib.metadata import version


def test_installed_package_exposes_its_version() -> None:
    import myclaw

    assert myclaw.__version__ == "0.1.0"
    assert version("myclaw") == myclaw.__version__
