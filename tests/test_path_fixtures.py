from pathlib import Path


def test_agent_home_fixture_redirects_the_fixed_home_without_precreating_it(
    agent_home: Path,
) -> None:
    assert agent_home == Path.home() / ".myclaw"
    assert agent_home.is_absolute()
    assert not agent_home.exists()


def test_workspace_fixture_provides_an_existing_normalized_directory(
    workspace: Path,
) -> None:
    assert workspace.is_absolute()
    assert workspace == workspace.resolve()
    assert workspace.is_dir()
