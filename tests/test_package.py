from hashlib import sha256
from importlib.metadata import distribution, version


def test_installed_package_exposes_its_version() -> None:
    import myclaw

    assert myclaw.__version__ == "0.1.0"
    assert version("myclaw") == myclaw.__version__


def test_installed_distribution_declares_and_bundles_apache_2_license() -> None:
    installed = distribution("myclaw")

    assert installed.metadata.get("License-Expression") == "Apache-2.0"
    assert installed.metadata.get_all("License-File") == ["LICENSE"]

    assert installed.files is not None
    license_paths = tuple(
        path for path in installed.files if path.parts[-2:] == ("licenses", "LICENSE")
    )
    assert len(license_paths) == 1
    license_text = installed.locate_file(license_paths[0]).read_text()
    normalized = license_text.replace("\r\n", "\n").strip() + "\n"
    assert sha256(normalized.encode()).hexdigest() == (
        "50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a"
    )
