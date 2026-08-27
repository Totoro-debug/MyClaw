from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import pytest

import myclaw.terminal.cli as cli
from myclaw.agent.loop import (
    AgentLoop,
    ForegroundConversationProjection,
    TerminalAgentLoopControl,
)
from myclaw.agent.message_bus import MessageBus
from myclaw.agent.runtime import RuntimeGenerationPresentation, RuntimeHost
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader, UserConfiguration
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import FatalManagementError
from myclaw.provider.factory import create_provider
from myclaw.skills.catalog import SkillMetadata
from tests.configuration.test_config import VALID_CONFIG


def _configuration(agent_home: AgentHome) -> UserConfiguration:
    agent_home.initialize()
    (agent_home.path / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    return ConfigLoader(agent_home).load()


class _CliContractApp:
    fatal_management_error: FatalManagementError | None = None

    def __init__(
        self,
        *,
        bus: MessageBus,
        control: TerminalAgentLoopControl,
        management_dispatcher: ManagementCommandDispatcher,
        skill_metadata: tuple[SkillMetadata, ...],
    ) -> None:
        del skill_metadata
        self._bus = bus
        self._initial_loop = cast(AgentLoop, control)
        self._control = control
        self._dispatcher = management_dispatcher
        self.events: list[str] = []
        self.target_loop: AgentLoop | None = None

    async def run_async(self) -> None:
        old = self._initial_loop
        old.session.add_message("user", "Persisted before same-Session replacement.")
        old.session.persist()
        await old.session.wait_for_pending_persist()
        old_session_id = old.session.session_id
        shared_bus = self._bus

        result = await self._dispatcher.resume(old_session_id)

        assert result.resumed_session_id == old_session_id
        assert self._bus is shared_bus
        target = cast(AgentLoop, self._control)
        assert target is not old
        assert target.session.session_id == old_session_id
        assert target._consumer_task is not None
        self.target_loop = target

    async def quiesce_for_rebind(self) -> None:
        self.events.append("quiesce")

    async def rebind_agent_loop(
        self,
        *,
        control: TerminalAgentLoopControl,
        skill_metadata: tuple[SkillMetadata, ...],
        session_projection: ForegroundConversationProjection,
    ) -> None:
        del skill_metadata
        self.events.append("rebind")
        unavailable = await self._dispatcher.dispatch("/status")
        assert unavailable.output == "route_unavailable: Runtime Generation is unavailable."
        target = cast(AgentLoop, control)
        assert target._consumer_task is None
        assert session_projection.session_id == target.session.session_id
        self._control = control


async def _exercise_cli_contract(
    *,
    agent_home: AgentHome,
    workspace: Path,
    configuration: UserConfiguration,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], AgentLoop]:
    app: _CliContractApp | None = None

    class ContractApp(_CliContractApp):
        def __init__(self, **kwargs: Any) -> None:
            nonlocal app
            super().__init__(**kwargs)
            app = self

    monkeypatch.setattr(cli, "TerminalConversationApp", ContractApp)
    await cli._run_cli_conversation(
        agent_home=agent_home,
        workspace=workspace,
        configuration=configuration,
    )

    assert app is not None
    assert app.target_loop is not None
    return app.events, app.target_loop


async def _exercise_legacy_contract(
    *,
    agent_home: AgentHome,
    workspace: Path,
    configuration: UserConfiguration,
) -> tuple[list[str], AgentLoop]:
    host = RuntimeHost(
        agent_home=agent_home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=create_provider,
        now=lambda: datetime.now().astimezone(),
        new_uuid=uuid4,
        timezone_name="Asia/Shanghai",
    )
    events: list[str] = []
    old = host.generation.agent_loop
    old.session.add_message("user", "Persisted before same-Session replacement.")
    old.session.persist()
    await old.session.wait_for_pending_persist()
    old_session_id = old.session.session_id
    shared_bus = host.bus

    async def quiesce() -> None:
        events.append("quiesce")

    async def rebind(presentation: RuntimeGenerationPresentation) -> None:
        events.append("rebind")
        unavailable = await host.management_dispatcher.dispatch("/status")
        assert unavailable.output == "route_unavailable: Runtime Generation is unavailable."
        target = cast(AgentLoop, presentation.control)
        assert target._consumer_task is None
        assert presentation.session_projection.session_id == old_session_id

    host.bind_terminal(rebind, quiesce=quiesce)
    try:
        await host.start()
        result = await host.management_dispatcher.resume(old_session_id)
        target = host.generation.agent_loop

        assert result.resumed_session_id == old_session_id
        assert host.bus is shared_bus
        assert target is not old
        assert target.session.session_id == old_session_id
        assert target._consumer_task is not None
        return events, target
    finally:
        host.unbind_terminal(rebind)
        await host.close()


@pytest.mark.parametrize("implementation", ("cli", "legacy"))
@pytest.mark.asyncio
async def test_cli_and_legacy_share_same_session_replacement_release_contract(
    implementation: Literal["cli", "legacy"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_home = AgentHome(tmp_path / f"{implementation}-agent-home")
    configuration = _configuration(agent_home)
    workspace = tmp_path / f"{implementation}-workspace"
    workspace.mkdir()
    constructed: list[AgentLoop] = []
    original_init = AgentLoop.__init__

    def recording_init(loop: AgentLoop, *args: Any, **kwargs: Any) -> None:
        original_init(loop, *args, **kwargs)
        constructed.append(loop)

    monkeypatch.setattr(AgentLoop, "__init__", recording_init)

    exercise: Callable[..., Awaitable[tuple[list[str], AgentLoop]]]
    if implementation == "cli":
        exercise = _exercise_cli_contract
        events, target = await exercise(
            agent_home=agent_home,
            workspace=workspace,
            configuration=configuration,
            monkeypatch=monkeypatch,
        )
    else:
        exercise = _exercise_legacy_contract
        events, target = await exercise(
            agent_home=agent_home,
            workspace=workspace,
            configuration=configuration,
        )

    assert events == ["quiesce", "rebind"]
    assert len(constructed) == 2
    assert constructed == [constructed[0], target]
