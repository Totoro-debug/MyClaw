from pathlib import Path

from myclaw.agent_home import AgentHome


def test_production_agent_home_is_fixed(agent_home: Path) -> None:
    assert AgentHome.production().path == agent_home


def test_first_initialization_creates_only_base_state(agent_home: Path) -> None:
    AgentHome(agent_home).initialize()

    tree = tuple(
        sorted(
            path.relative_to(agent_home).as_posix() + ("/" if path.is_dir() else "")
            for path in agent_home.rglob("*")
        )
    )
    assert (tree, (agent_home / "memory" / "memory.md").read_bytes()) == (
        ("memory/", "memory/memory.md", "sessions/"),
        (
            b"# Long-term Memory\n\n"
            b"## User Info\n\n"
            b"## User Preference\n\n"
            b"## Project Fact\n\n"
            b"## Lesson\n"
        ),
    )


def test_repeated_initialization_preserves_long_term_memory(agent_home: Path) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    memory = agent_home / "memory" / "memory.md"
    existing_content = b"# Personal memory\n\nKeep this exact content.\n"
    memory.write_bytes(existing_content)

    home.initialize()

    assert memory.read_bytes() == existing_content
