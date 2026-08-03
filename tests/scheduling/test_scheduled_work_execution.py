import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolModelMessage,
)
from myclaw.schedule.background_coordination import RuntimeEventBroker, ScheduledWorkCoordinator
from myclaw.schedule.records import ScheduledWork
from myclaw.schedule.scheduled_work_execution import (
    ScheduledWorkModelSettings,
    ScheduledWorkRunner,
)
from myclaw.session.records import (
    AssistantSessionMessage,
    ConversationSession,
    ToolSessionMessage,
    UserSessionMessage,
)
from myclaw.session.session_store import JsonlSessionStore
from myclaw.tools.base import BaseTool
from myclaw.tools.files.file_tools import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from myclaw.tools.models import (
    ModelToolCall,
    ToolResult,
)
from myclaw.tools.security import Security
from myclaw.tools.shell.shell_policy import ShellRequest
from myclaw.tools.shell.shell_tool import ShellBoundary, ShellTool
from myclaw.tools.tool_artifacts import externalize_tool_result
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, persist_scheduled_work

LOCAL_TIMEZONE = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 12, 23, 0, 0, 123456, tzinfo=LOCAL_TIMEZONE)
TASK_ID = "550e8400-e29b-41d4-a716-446655440000"
TASK_SESSION_ID = "20260712-220000-123000_0f8fad5b-d9cb-469f-a165-70867728950e"
USER_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
REQUEST_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")
ASSISTANT_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
USER_TWO_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
REQUEST_TWO_UUID = UUID("a8098c1a-f86e-4f33-8a28-25f602f8e603")
ASSISTANT_TWO_UUID = UUID("67e55044-10b1-426f-9247-bb680e5fe0c8")
FINAL_RUNTIME_UUID = UUID("33333333-3333-4333-8333-333333333333")


def _task() -> ScheduledWork:
    return ScheduledWork(
        id=TASK_ID,
        title="Weekly project review",
        cron="0 9 * * 1",
        prompt="Review the current project and summarize open risks.",
        created_at=NOW - timedelta(hours=1),
        enabled=True,
        session_id=TASK_SESSION_ID,
    )


def _usage() -> ModelUsage:
    return ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15)


def _externalizer_for(
    *, workspace: Path, max_tool_result_chars: int
) -> Callable[[str], Callable[[ToolResult], ToolResult]]:
    workspace_state = WorkspaceState(Workspace.from_path(workspace))
    workspace_state.initialize(agent_home_root=Path.home() / ".myclaw")

    def for_session(session_id: str) -> Callable[[ToolResult], ToolResult]:
        def externalize(result: ToolResult) -> ToolResult:
            return externalize_tool_result(
                result,
                workspace_state=workspace_state,
                session_id=session_id,
                max_tool_result_chars=max_tool_result_chars,
            )

        return externalize

    return for_session


class BlockingShellBoundary:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._release = asyncio.Event()
        self.cancelled = False

    async def execute(self, request: ShellRequest) -> str:
        del request
        self.started.set()
        try:
            await self._release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "released"


class FailingShellBoundary:
    async def execute(self, request: ShellRequest) -> str:
        del request
        raise OSError("private subprocess failure")


class ObservingToolGateway(ToolGateway):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[ToolResult] = []

    async def call(self, tool_call: ModelToolCall) -> ToolResult:
        result = await super().call(tool_call)
        self.results.append(result)
        return result


def _tool_gateway(
    *,
    agent_home: Path,
    workspace_state: WorkspaceState,
    session_id: str,
    shell: ShellBoundary | None = None,
    gateway: ToolGateway | None = None,
) -> ToolGateway:
    workspace_identity = workspace_state.workspace
    workspace = Path(workspace_identity.path)
    security = Security(
        workspace=workspace_identity,
        agent_home=agent_home,
        artifact_directory=workspace_state.sessions_directory / "artifacts" / session_id,
    )
    tools: list[BaseTool] = [
        ReadFileTool(security=security),
        ListFilesTool(security=security),
        SearchFilesTool(security=security),
        WriteFileTool(),
        EditFileTool(),
    ]
    if shell is not None:
        tools.append(ShellTool(workspace=workspace, boundary=shell))
    result = ToolGateway() if gateway is None else gateway
    result.register_tools(tuple(tools))
    return result


def _long_path(path: Path) -> Path:
    return HOST_FILESYSTEM.path_for_io(path)


def _artifact_directory(*, workspace: Path, session_id: str) -> Path:
    state = WorkspaceState(Workspace.from_path(workspace))
    return state.sessions_directory / "artifacts" / session_id


class LoadFailingSessionStore(JsonlSessionStore):
    def __init__(
        self,
        *,
        workspace_state: WorkspaceState,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        fail_on_load: int,
    ) -> None:
        super().__init__(
            workspace_state=workspace_state,
            now=now,
            new_uuid=new_uuid,
        )
        self._fail_on_load = fail_on_load
        self._load_calls = 0

    async def load(self, session_id: str) -> ConversationSession:
        self._load_calls += 1
        if self._load_calls == self._fail_on_load:
            raise OSError("private load failure")
        return await super().load(session_id)


class ToolAppendFailingSessionStore(JsonlSessionStore):
    async def append_message(
        self,
        session_id: str,
        message: AssistantSessionMessage | ToolSessionMessage | UserSessionMessage,
    ) -> None:
        if isinstance(message, ToolSessionMessage):
            raise OSError("injected pre-write Tool append failure")
        await super().append_message(session_id, message)


class IndeterminateToolAppendSessionStore(JsonlSessionStore):
    def __init__(
        self,
        *,
        workspace_state: WorkspaceState,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
    ) -> None:
        super().__init__(
            workspace_state=workspace_state,
            now=now,
            new_uuid=new_uuid,
        )
        self._fail_reconciliation_load = False

    async def append_message(
        self,
        session_id: str,
        message: AssistantSessionMessage | ToolSessionMessage | UserSessionMessage,
    ) -> None:
        if isinstance(message, ToolSessionMessage):
            self._fail_reconciliation_load = True
            raise OSError("injected indeterminate Tool append failure")
        await super().append_message(session_id, message)

    async def load(self, session_id: str) -> ConversationSession:
        if self._fail_reconciliation_load:
            self._fail_reconciliation_load = False
            raise OSError("injected reconciliation load failure")
        return await super().load(session_id)


class BlockingToolAppendSessionStore(JsonlSessionStore):
    def __init__(
        self,
        *,
        workspace_state: WorkspaceState,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
    ) -> None:
        super().__init__(
            workspace_state=workspace_state,
            now=now,
            new_uuid=new_uuid,
        )
        self.tool_append_started = asyncio.Event()

    async def append_message(
        self,
        session_id: str,
        message: AssistantSessionMessage | ToolSessionMessage | UserSessionMessage,
    ) -> None:
        if isinstance(message, ToolSessionMessage):
            self.tool_append_started.set()
            await asyncio.Future()
        await super().append_message(session_id, message)


class EffectThenBlockToolAppendSessionStore(JsonlSessionStore):
    def __init__(
        self,
        *,
        workspace_state: WorkspaceState,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
    ) -> None:
        super().__init__(
            workspace_state=workspace_state,
            now=now,
            new_uuid=new_uuid,
        )
        self.tool_message_written = asyncio.Event()

    async def append_message(
        self,
        session_id: str,
        message: AssistantSessionMessage | ToolSessionMessage | UserSessionMessage,
    ) -> None:
        await super().append_message(session_id, message)
        if isinstance(message, ToolSessionMessage):
            self.tool_message_written.set()
            await asyncio.Future()


class ToolPublicationBaseError(BaseException):
    pass


class EffectThenRaiseToolAppendSessionStore(JsonlSessionStore):
    def __init__(
        self,
        *,
        failure: BaseException,
        workspace_state: WorkspaceState,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
    ) -> None:
        super().__init__(
            workspace_state=workspace_state,
            now=now,
            new_uuid=new_uuid,
        )
        self._failure = failure

    async def append_message(
        self,
        session_id: str,
        message: AssistantSessionMessage | ToolSessionMessage | UserSessionMessage,
    ) -> None:
        await super().append_message(session_id, message)
        if isinstance(message, ToolSessionMessage):
            raise self._failure


@pytest.mark.asyncio
async def test_first_scheduled_work_trigger_persists_a_complete_cron_turn(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    long_term_memory = "# Long-term Memory\n\n## Project Fact\n\nUse strict TDD.\n"
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="No open risks were found."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    task = _task()
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory=long_term_memory,
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )
    assert not sessions.path_for(task.session_id).exists()

    await runner.run(task)

    session = await sessions.load(task.session_id)
    assert session.metadata.id == task.session_id
    assert session.metadata.title == task.title
    assert session.metadata.created_at == task.created_at.replace(microsecond=123000)
    assert len(session.messages) == 2
    user, assistant = session.messages
    assert isinstance(user, UserSessionMessage)
    assert user.content == task.prompt
    assert isinstance(assistant, AssistantSessionMessage)
    assert assistant.content == "No open risks were found."
    assert assistant.status == "completed"
    assert assistant.usage == _usage()

    assert len(provider.complete_requests) == 1
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert request.route == "cron"
    assert request.stream is False
    assert request.model == "cron-model"
    assert str(Workspace.from_path(workspace).path) in request.system_prompt
    assert long_term_memory in request.system_prompt
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert task.prompt in request.messages[0].content
    assert "<runtime_context>" in request.messages[0].content
    assert f"session_id: {task.session_id}" in request.messages[0].content


@pytest.mark.asyncio
async def test_repeated_scheduled_work_triggers_reuse_the_task_session_history(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="First scheduled result."),
                usage=_usage(),
                finish_reason="stop",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Second scheduled result."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter(
            (
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )
    task = _task()

    await runner.run(task)
    await runner.run(task)

    session = await sessions.load(task.session_id)
    assert [message.role for message in session.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    second_request = provider.complete_requests[1]
    assert isinstance(second_request, ModelRequest)
    assert [message.role for message in second_request.messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert second_request.messages[0].content == task.prompt
    assert second_request.messages[1].content == "First scheduled result."
    assert "<runtime_context>" in second_request.messages[2].content
    assert task.prompt in second_request.messages[2].content


@pytest.mark.asyncio
async def test_scheduled_work_auto_refuses_ask_tools_and_completes_the_agent_turn(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    call = ModelToolCall(
        id="call_write",
        name="write_file",
        arguments='{"path":"scheduled.txt","content":"must not be written"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(
                    content="I need to write a file.",
                    tool_calls=(call,),
                ),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="The write required confirmation."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    task = _task()
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter(
            (
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )

    await runner.run(task)

    assert not (workspace / "scheduled.txt").exists()
    session = await sessions.load(task.session_id)
    assert [message.role for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_result = session.messages[2]
    assert isinstance(tool_result, ToolSessionMessage)
    assert tool_result.tool_call_id == call.id
    assert tool_result.status == "refused"
    assert len(provider.complete_requests) == 2
    follow_up = provider.complete_requests[1]
    assert isinstance(follow_up, ModelRequest)
    assert [message.role for message in follow_up.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    follow_up_tool = follow_up.messages[-1]
    assert isinstance(follow_up_tool, ToolModelMessage)
    assert follow_up_tool.tool_call_id == call.id
    assert follow_up_tool.content == tool_result.content


@pytest.mark.asyncio
async def test_scheduled_work_commits_a_published_oversized_tool_result(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    raw_result = "SCHEDULED OVERSIZED TOOL RESULT"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    call = ModelToolCall(
        id="call_scheduled_artifact",
        name="read_file",
        arguments='{"path":"large.txt"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Inspecting.", tool_calls=(call,)),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Inspection complete."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    gateway = ObservingToolGateway()
    _tool_gateway(
        agent_home=agent_home,
        workspace_state=sessions.workspace_state,
        session_id=TASK_SESSION_ID,
        gateway=gateway,
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter(
            (
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
        tool_gateway_for=lambda _: gateway,
        externalize_result_for=_externalizer_for(
            workspace=workspace,
            max_tool_result_chars=1,
        ),
    )

    outcome = await runner.run(_task())

    assert outcome.status == "completed"
    session = await sessions.load(TASK_SESSION_ID)
    assert [message.role for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_message = session.messages[2]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.artifact is not None
    artifact_path = _long_path(
        _artifact_directory(workspace=workspace, session_id=TASK_SESSION_ID)
        / "call_scheduled_artifact.txt"
    )
    assert artifact_path.read_text(encoding="utf-8") == raw_result
    assert artifact_path.read_text(encoding="utf-8") == raw_result


@pytest.mark.asyncio
async def test_model_failure_is_persisted_without_stopping_the_next_scheduled_work(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(ErrorInfo(code="model_failed", message="Safe cron failure.")),
            ModelResponse(
                message=AssistantModelMessage(content="The other task completed."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter(
            (
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )
    failed_task = _task()
    completed_task = replace(
        failed_task,
        id="11111111-1111-4111-8111-111111111111",
        title="Daily status",
        prompt="Summarize today's status.",
        session_id="20260712-220000-123000_22222222-2222-4222-8222-222222222222",
    )

    failed = await runner.run(failed_task)
    completed = await runner.run(completed_task)

    assert failed.status == "failed"
    assert failed.content == ""
    assert failed.error == ErrorInfo(code="model_failed", message="Safe cron failure.")
    assert completed.status == "completed"
    assert completed.content == "The other task completed."
    assert completed.error is None
    failed_session = await sessions.load(failed_task.session_id)
    failed_assistant = failed_session.messages[-1]
    assert isinstance(failed_assistant, AssistantSessionMessage)
    assert failed_assistant.status == "error"
    assert failed_assistant.error is not None
    assert failed_assistant.error.code == "model_failed"
    assert failed_assistant.error.message == "Safe cron failure."
    completed_session = await sessions.load(completed_task.session_id)
    assert completed_session.messages[-1].role == "assistant"


@pytest.mark.asyncio
async def test_scheduled_work_records_one_terminal_failure_with_task_session_context(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    task = replace(
        _task(),
        title="PRIVATE TASK DEFINITION TITLE",
        cron="17 4 * * 2",
        prompt="PRIVATE TASK PROMPT",
    )
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(ErrorInfo(code="model_failed", message="PRIVATE MODEL OUTPUT")),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="PRIVATE LONG-TERM MEMORY",
        settings=ScheduledWorkModelSettings(
            model="PRIVATE MODEL ID",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )
    events = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=events,
        now=lambda: NOW,
        new_uuid=uuid4,
        workspace_state=state,
    )

    result = await coordinator.trigger(task)
    await events.next_background_event()

    assert result.status == "failed"
    assert result.error == ErrorInfo(
        code="model_failed",
        message="PRIVATE MODEL OUTPUT",
    )
    content = (state.logs_directory / f"{task.session_id}.log").read_text(encoding="utf-8")
    assert content.count(" ERROR ") == 1
    assert "myclaw.schedule.scheduled_work_execution:run:" in content
    assert "Scheduled Work failed code=model_failed" in content
    assert "Traceback (most recent call last):" in content
    assert "ModelCallError: PRIVATE MODEL OUTPUT" in content
    for private_content in (
        "PRIVATE TASK DEFINITION TITLE",
        "PRIVATE TASK PROMPT",
        "PRIVATE LONG-TERM MEMORY",
        "PRIVATE MODEL ID",
        task.id,
        task.cron,
    ):
        assert private_content not in content


@pytest.mark.asyncio
async def test_scheduled_work_records_an_unhandled_terminal_exception_once(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    provider = ScriptedFakeProvider(completions=(RuntimeError("technical cron execution failure"),))
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID)).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=RuntimeEventBroker(),
        now=lambda: NOW,
        new_uuid=uuid4,
        workspace_state=state,
    )

    with pytest.raises(RuntimeError, match="technical cron execution failure"):
        await coordinator.trigger(_task())

    content = (state.logs_directory / f"{TASK_SESSION_ID}.log").read_text(encoding="utf-8")
    assert content.count(" ERROR ") == 1
    assert "myclaw.schedule.scheduled_work_execution:run:" in content
    assert "Scheduled Work crashed" in content
    assert "RuntimeError: technical cron execution failure" in content
    assert _task().prompt not in content


@pytest.mark.asyncio
async def test_session_publication_failure_is_isolated_from_the_next_scheduled_work(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    publication_calls = 0

    def fail_first_publication(path: Path, content: bytes) -> None:
        nonlocal publication_calls
        publication_calls += 1
        if publication_calls == 1:
            raise OSError("private disk failure detail")
        HOST_FILESYSTEM.atomic_replace_bytes(path, content)

    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
        replace_bytes=fail_first_publication,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="First result was fully generated."),
                usage=_usage(),
                finish_reason="stop",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Second task remained isolated."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter(
            (
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )
    failed_task = _task()
    completed_task = replace(
        failed_task,
        id="11111111-1111-4111-8111-111111111111",
        title="Independent task",
        prompt="Run independently.",
        session_id="20260712-220000-123000_22222222-2222-4222-8222-222222222222",
    )
    events = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=events,
        now=lambda: NOW,
        new_uuid=uuid4,
        workspace_state=state,
    )

    failed = await coordinator.trigger(failed_task)
    completed = await coordinator.trigger(completed_task)
    await events.next_background_event()
    await events.next_background_event()

    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.code == "persistence_error"
    assert "private disk failure detail" not in failed.error.message
    assert completed.status == "completed"
    first_session = await sessions.load(failed_task.session_id)
    assert [message.role for message in first_session.messages] == ["user", "assistant"]
    assert first_session.messages[-1].content == "First result was fully generated."
    second_session = await sessions.load(completed_task.session_id)
    assert second_session.messages[-1].content == "Second task remained isolated."
    content = (state.logs_directory / f"{failed_task.session_id}.log").read_text(
        encoding="utf-8"
    )
    assert content.count(" ERROR ") == 1
    assert "myclaw.schedule.scheduled_work_execution:run:" in content
    assert "Scheduled Work failed code=persistence_error" in content
    assert "OSError: private disk failure detail" in content
    assert failed_task.prompt not in content
    assert "First result was fully generated." not in content


@pytest.mark.asyncio
async def test_corrupt_task_session_is_isolated_before_the_model_call(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    failed_task = _task()
    corrupt_path = sessions.path_for(failed_task.session_id)
    io_path = HOST_FILESYSTEM.path_for_io(corrupt_path)
    io_path.parent.mkdir(parents=True)
    io_path.write_text("{not valid session json}\n", encoding="utf-8")
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Independent task completed."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, USER_TWO_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )
    completed_task = replace(
        failed_task,
        id="11111111-1111-4111-8111-111111111111",
        title="Independent task",
        prompt="Run independently.",
        session_id="20260712-220000-123000_22222222-2222-4222-8222-222222222222",
    )

    failed = await runner.run(failed_task)
    completed = await runner.run(completed_task)

    assert failed.status == "failed"
    assert failed.error == ErrorInfo(
        code="persistence_error",
        message="Scheduled Work Session could not be updated.",
    )
    assert completed.status == "completed"
    assert len(provider.complete_requests) == 1
    assert io_path.read_text(encoding="utf-8") == "{not valid session json}\n"


@pytest.mark.asyncio
async def test_tool_result_publication_failure_is_isolated_from_the_next_task(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    publication_calls = 0

    def fail_tool_publication(path: Path, content: bytes) -> None:
        nonlocal publication_calls
        publication_calls += 1
        if publication_calls == 2:
            raise OSError("private tool publication failure")
        HOST_FILESYSTEM.atomic_replace_bytes(path, content)

    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
        replace_bytes=fail_tool_publication,
    )
    call = ModelToolCall(
        id="call_refused_write",
        name="write_file",
        arguments='{"path":"blocked.txt","content":"must not be written"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Trying a write.", tool_calls=(call,)),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Independent task completed."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter(
            (
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
                FINAL_RUNTIME_UUID,
            )
        ).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )
    failed_task = _task()
    completed_task = replace(
        failed_task,
        id="11111111-1111-4111-8111-111111111111",
        title="Independent task",
        prompt="Run independently.",
        session_id="20260712-220000-123000_22222222-2222-4222-8222-222222222222",
    )

    failed = await runner.run(failed_task)
    completed = await runner.run(completed_task)

    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.code == "persistence_error"
    assert "private tool publication failure" not in failed.error.message
    assert completed.status == "completed"
    failed_session = await sessions.load(failed_task.session_id)
    assert [message.role for message in failed_session.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert not (workspace / "blocked.txt").exists()


@pytest.mark.asyncio
async def test_scheduled_work_commits_an_artifact_after_effect_then_raise_publication(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    publication_calls = 0

    def fail_tool_metadata_publication(path: Path, content: bytes) -> None:
        nonlocal publication_calls
        publication_calls += 1
        if publication_calls == 2:
            raise OSError("injected metadata publication failure")
        HOST_FILESYSTEM.atomic_replace_bytes(path, content)

    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
        replace_bytes=fail_tool_metadata_publication,
    )
    raw_result = "DURABLE SCHEDULED ARTIFACT"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    call = ModelToolCall(
        id="call_effect_then_raise_artifact",
        name="read_file",
        arguments='{"path":"large.txt"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Inspecting.", tool_calls=(call,)),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
        )
    )
    gateway = ObservingToolGateway()
    _tool_gateway(
        agent_home=agent_home,
        workspace_state=sessions.workspace_state,
        session_id=TASK_SESSION_ID,
        gateway=gateway,
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID, USER_TWO_UUID)).__next__,
        tool_gateway_for=lambda _: gateway,
        externalize_result_for=_externalizer_for(
            workspace=workspace,
            max_tool_result_chars=1,
        ),
    )

    outcome = await runner.run(_task())

    assert outcome.status == "failed"
    assert outcome.error == ErrorInfo(
        code="persistence_error",
        message="Scheduled Work Session could not be updated.",
    )
    session = await sessions.load(TASK_SESSION_ID)
    assert [message.role for message in session.messages] == ["user", "assistant", "tool"]
    tool_message = session.messages[-1]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.id == str(USER_TWO_UUID)
    assert tool_message.artifact is not None
    artifact_path = _long_path(
        _artifact_directory(workspace=workspace, session_id=TASK_SESSION_ID)
        / "call_effect_then_raise_artifact.txt"
    )
    assert artifact_path.read_text(encoding="utf-8") == raw_result
    assert artifact_path.read_text(encoding="utf-8") == raw_result


@pytest.mark.asyncio
async def test_scheduled_work_leaves_an_orphan_artifact_when_tool_message_was_not_written(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = ToolAppendFailingSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    raw_result = "UNPUBLISHED SCHEDULED ARTIFACT"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    call = ModelToolCall(
        id="call_unpublished_artifact",
        name="read_file",
        arguments='{"path":"large.txt"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Inspecting.", tool_calls=(call,)),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
        )
    )
    gateway = ObservingToolGateway()
    _tool_gateway(
        agent_home=agent_home,
        workspace_state=sessions.workspace_state,
        session_id=TASK_SESSION_ID,
        gateway=gateway,
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID, USER_TWO_UUID)).__next__,
        tool_gateway_for=lambda _: gateway,
        externalize_result_for=_externalizer_for(
            workspace=workspace,
            max_tool_result_chars=1,
        ),
    )

    outcome = await runner.run(_task())

    assert outcome.status == "failed"
    assert outcome.error is not None
    assert outcome.error.code == "persistence_error"
    session = await sessions.load(TASK_SESSION_ID)
    assert [message.role for message in session.messages] == ["user", "assistant"]
    artifact_path = _long_path(
        _artifact_directory(workspace=workspace, session_id=TASK_SESSION_ID)
        / "call_unpublished_artifact.txt"
    )
    assert artifact_path.read_text(encoding="utf-8") == raw_result


@pytest.mark.asyncio
async def test_scheduled_work_preserves_an_artifact_when_publication_is_indeterminate(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = IndeterminateToolAppendSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    raw_result = "INDETERMINATE SCHEDULED ARTIFACT"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    call = ModelToolCall(
        id="call_indeterminate_artifact",
        name="read_file",
        arguments='{"path":"large.txt"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Inspecting.", tool_calls=(call,)),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
        )
    )
    gateway = ObservingToolGateway()
    _tool_gateway(
        agent_home=agent_home,
        workspace_state=sessions.workspace_state,
        session_id=TASK_SESSION_ID,
        gateway=gateway,
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID, USER_TWO_UUID)).__next__,
        tool_gateway_for=lambda _: gateway,
        externalize_result_for=_externalizer_for(
            workspace=workspace,
            max_tool_result_chars=1,
        ),
    )

    outcome = await runner.run(_task())

    assert outcome.status == "failed"
    assert outcome.error is not None
    assert outcome.error.code == "persistence_error"
    with pytest.raises(OSError, match="injected reconciliation load failure"):
        await sessions.load(TASK_SESSION_ID)
    session = await sessions.load(TASK_SESSION_ID)
    assert [message.role for message in session.messages] == ["user", "assistant"]
    artifact_path = _long_path(
        _artifact_directory(workspace=workspace, session_id=TASK_SESSION_ID)
        / "call_indeterminate_artifact.txt"
    )
    assert artifact_path.read_text(encoding="utf-8") == raw_result
    assert artifact_path.read_text(encoding="utf-8") == raw_result


@pytest.mark.asyncio
async def test_cancelling_scheduled_tool_publication_leaves_an_orphan_artifact(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = BlockingToolAppendSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    raw_result = "CANCELLED SCHEDULED ARTIFACT"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    call = ModelToolCall(
        id="call_cancelled_artifact",
        name="read_file",
        arguments='{"path":"large.txt"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Inspecting.", tool_calls=(call,)),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
        )
    )
    gateway = ObservingToolGateway()
    _tool_gateway(
        agent_home=agent_home,
        workspace_state=sessions.workspace_state,
        session_id=TASK_SESSION_ID,
        gateway=gateway,
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID, USER_TWO_UUID)).__next__,
        tool_gateway_for=lambda _: gateway,
        externalize_result_for=_externalizer_for(
            workspace=workspace,
            max_tool_result_chars=1,
        ),
    )
    execution = asyncio.create_task(runner.run(_task()))
    await asyncio.wait_for(sessions.tool_append_started.wait(), timeout=1)

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    session = await sessions.load(TASK_SESSION_ID)
    assert [message.role for message in session.messages] == ["user", "assistant"]
    artifact_path = _long_path(
        _artifact_directory(workspace=workspace, session_id=TASK_SESSION_ID)
        / "call_cancelled_artifact.txt"
    )
    assert artifact_path.read_text(encoding="utf-8") == raw_result


@pytest.mark.asyncio
async def test_cancelling_after_scheduled_tool_publication_commits_the_durable_artifact(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = EffectThenBlockToolAppendSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    raw_result = "DURABLE CANCELLED SCHEDULED ARTIFACT"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    call = ModelToolCall(
        id="call_effect_then_cancel_artifact",
        name="read_file",
        arguments='{"path":"large.txt"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Inspecting.", tool_calls=(call,)),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
        )
    )
    gateway = ObservingToolGateway()
    _tool_gateway(
        agent_home=agent_home,
        workspace_state=sessions.workspace_state,
        session_id=TASK_SESSION_ID,
        gateway=gateway,
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID, USER_TWO_UUID)).__next__,
        tool_gateway_for=lambda _: gateway,
        externalize_result_for=_externalizer_for(
            workspace=workspace,
            max_tool_result_chars=1,
        ),
    )
    execution = asyncio.create_task(runner.run(_task()))
    await asyncio.wait_for(sessions.tool_message_written.wait(), timeout=1)

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    session = await sessions.load(TASK_SESSION_ID)
    assert [message.role for message in session.messages] == ["user", "assistant", "tool"]
    tool_message = session.messages[-1]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.artifact is not None
    artifact_path = _long_path(
        _artifact_directory(workspace=workspace, session_id=TASK_SESSION_ID)
        / "call_effect_then_cancel_artifact.txt"
    )
    assert artifact_path.read_text(encoding="utf-8") == raw_result


@pytest.mark.asyncio
async def test_scheduled_tool_publication_preserves_a_primary_base_exception(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    failure = ToolPublicationBaseError("primary publication failure")
    sessions = EffectThenRaiseToolAppendSessionStore(
        failure=failure,
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    raw_result = "DURABLE BASE EXCEPTION ARTIFACT"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    call = ModelToolCall(
        id="call_base_exception_artifact",
        name="read_file",
        arguments='{"path":"large.txt"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Inspecting.", tool_calls=(call,)),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
        )
    )
    gateway = ObservingToolGateway()
    _tool_gateway(
        agent_home=agent_home,
        workspace_state=sessions.workspace_state,
        session_id=TASK_SESSION_ID,
        gateway=gateway,
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID, USER_TWO_UUID)).__next__,
        tool_gateway_for=lambda _: gateway,
        externalize_result_for=_externalizer_for(
            workspace=workspace,
            max_tool_result_chars=1,
        ),
    )

    with pytest.raises(ToolPublicationBaseError) as raised:
        await runner.run(_task())

    assert raised.value is failure
    session = await sessions.load(TASK_SESSION_ID)
    assert [message.role for message in session.messages] == ["user", "assistant", "tool"]
    tool_message = session.messages[-1]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.artifact is not None
    artifact_path = _long_path(
        _artifact_directory(workspace=workspace, session_id=TASK_SESSION_ID)
        / "call_base_exception_artifact.txt"
    )
    assert artifact_path.read_text(encoding="utf-8") == raw_result


@pytest.mark.asyncio
async def test_model_error_publication_failure_becomes_a_persistence_outcome(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    publication_calls = 0

    def fail_error_publication(path: Path, content: bytes) -> None:
        nonlocal publication_calls
        publication_calls += 1
        if publication_calls == 1:
            raise OSError("private model error publication failure")
        HOST_FILESYSTEM.atomic_replace_bytes(path, content)

    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
        replace_bytes=fail_error_publication,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(ErrorInfo(code="model_failed", message="Safe model failure.")),
            ModelResponse(
                message=AssistantModelMessage(content="Independent task completed."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter(
            (
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )
    failed_task = _task()
    completed_task = replace(
        failed_task,
        id="11111111-1111-4111-8111-111111111111",
        title="Independent task",
        prompt="Run independently.",
        session_id="20260712-220000-123000_22222222-2222-4222-8222-222222222222",
    )

    failed = await runner.run(failed_task)
    completed = await runner.run(completed_task)

    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.code == "persistence_error"
    assert "private model error publication failure" not in failed.error.message
    assert completed.status == "completed"
    failed_session = await sessions.load(failed_task.session_id)
    failed_assistant = failed_session.messages[-1]
    assert isinstance(failed_assistant, AssistantSessionMessage)
    assert failed_assistant.status == "error"
    assert failed_assistant.error is not None
    assert failed_assistant.error.code == "model_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fail_on_load", "include_tool_turn", "failed_roles", "provider_request_count"),
    [
        (1, False, ["user"], 1),
        (2, True, ["user", "assistant", "tool"], 2),
    ],
    ids=("first-load", "post-tool-load"),
)
async def test_session_load_failure_is_isolated_at_every_cron_loop_boundary(
    fail_on_load: int,
    include_tool_turn: bool,
    failed_roles: list[str],
    provider_request_count: int,
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = LoadFailingSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
        fail_on_load=fail_on_load,
    )
    call = ModelToolCall(
        id="call_load_boundary",
        name="write_file",
        arguments='{"path":"blocked.txt","content":"must not be written"}',
    )
    first_response = ModelResponse(
        message=AssistantModelMessage(content="Trying a write.", tool_calls=(call,)),
        usage=_usage(),
        finish_reason="tool_calls",
    )
    final_response = ModelResponse(
        message=AssistantModelMessage(content="Independent task completed."),
        usage=_usage(),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(
        completions=(first_response, final_response) if include_tool_turn else (final_response,)
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter(
            (
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
                FINAL_RUNTIME_UUID,
            )
        ).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )
    failed_task = _task()
    completed_task = replace(
        failed_task,
        id="11111111-1111-4111-8111-111111111111",
        title="Independent task",
        prompt="Run independently.",
        session_id="20260712-220000-123000_22222222-2222-4222-8222-222222222222",
    )

    failed = await runner.run(failed_task)
    completed = await runner.run(completed_task)

    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.code == "persistence_error"
    assert "private load failure" not in failed.error.message
    assert completed.status == "completed"
    assert len(provider.complete_requests) == provider_request_count
    failed_session = await sessions.load(failed_task.session_id)
    assert [message.role for message in failed_session.messages] == failed_roles


@pytest.mark.asyncio
async def test_cancelling_a_running_scheduled_tool_repairs_history_without_closing_provider(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    shell = BlockingShellBoundary()
    call = ModelToolCall(
        id="call_pwd",
        name="shell",
        arguments='{"command":"pwd","timeout":60}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(
                    content="Checking the Workspace.", tool_calls=(call,)
                ),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
        )
    )
    task = _task()
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID, USER_TWO_UUID)).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
            shell=shell,
        ),
    )
    execution = asyncio.create_task(runner.run(task))
    await asyncio.wait_for(shell.started.wait(), timeout=1)

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    assert shell.cancelled is True
    assert provider.closed is False
    session = await sessions.load(task.session_id)
    assert [message.role for message in session.messages] == ["user", "assistant", "tool"]
    cancelled_tool = session.messages[-1]
    assert isinstance(cancelled_tool, ToolSessionMessage)
    assert cancelled_tool.tool_call_id == call.id
    assert cancelled_tool.status == "error"
    assert cancelled_tool.content == "Scheduled Work tool call cancelled."


@pytest.mark.asyncio
async def test_scheduled_work_uses_a_normalized_task_specific_session_title(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Completed long-title task."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    long_title = "A" * 80
    task = replace(_task(), title=long_title)
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
        ),
    )

    result = await runner.run(task)

    assert result.status == "completed"
    session = await sessions.load(task.session_id)
    assert session.metadata.title == "A" * 60


@pytest.mark.asyncio
async def test_runtime_scheduled_work_uses_current_workspace_context_and_tool_catalog(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    workspace_memory = "# Workspace Long-term Memory\n\nCurrent cron context.\n"
    state.long_term_memory_path.write_text(workspace_memory, encoding="utf-8")
    legacy_memory = b"# Legacy Agent Home Memory\n"
    legacy_memory_path = agent_home / "memory" / "memory.md"
    legacy_memory_path.parent.mkdir(parents=True)
    legacy_memory_path.write_bytes(legacy_memory)
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace("max_tool_result_chars = 60000", "max_tool_result_chars = 1000"),
        encoding="utf-8",
    )
    raw_result = "W" * 2_000
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    call = ModelToolCall(
        id="call_runtime_scheduled_artifact",
        name="read_file",
        arguments='{"path":"large.txt"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Inspecting.", tool_calls=(call,)),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Runtime cron result."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
        shell=BlockingShellBoundary(),
    )
    task = _task()

    result = await runtime.scheduled_work_coordinator.trigger(task)

    assert result.status == "completed"
    assert runtime.session_id != task.session_id
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert request.route == "cron"
    assert request.model == "claude-model"
    tool_names = [schema["function"]["name"] for schema in request.tools]
    assert tool_names == [
        "read_file",
        "list_files",
        "search_files",
        "write_file",
        "edit_file",
        "shell",
        "create_scheduled_work",
    ]
    assert "- shell:" in request.system_prompt
    assert "- create_scheduled_work:" in request.system_prompt
    assert "web_search" not in request.system_prompt
    assert "web_fetch" not in request.system_prompt
    assert str(state.workspace.path) in request.system_prompt
    assert workspace_memory in request.system_prompt
    assert "Legacy Agent Home Memory" not in request.system_prompt
    session = await runtime.sessions.load(task.session_id)
    assert [message.role for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_message = session.messages[2]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.artifact is not None
    artifact_path = (
        state.sessions_directory
        / "artifacts"
        / task.session_id
        / "call_runtime_scheduled_artifact.txt"
    )
    assert artifact_path.read_text(encoding="utf-8") == raw_result
    assert runtime.sessions.path_for(task.session_id).parent == state.sessions_directory
    assert legacy_memory_path.read_bytes() == legacy_memory


@pytest.mark.asyncio
async def test_scheduled_tool_failure_is_safe_history_and_the_turn_can_finish(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    call = ModelToolCall(
        id="call_pwd_failure",
        name="shell",
        arguments='{"command":"pwd","timeout":60}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Checking.", tool_calls=(call,)),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="The check failed safely."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    task = _task()
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter(
            (
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
        tool_gateway_for=lambda session_id: _tool_gateway(
            agent_home=agent_home,
            workspace_state=sessions.workspace_state,
            session_id=session_id,
            shell=FailingShellBoundary(),
        ),
    )

    result = await runner.run(task)

    assert result.status == "completed"
    session = await sessions.load(task.session_id)
    tool_result = session.messages[2]
    assert isinstance(tool_result, ToolSessionMessage)
    assert tool_result.status == "error"
    assert "private subprocess failure" not in tool_result.content
    assert session.messages[-1].content == "The check failed safely."


@pytest.mark.asyncio
async def test_runtime_scheduled_work_refuses_recursive_task_creation(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    recursive_call = ModelToolCall(
        id="call_recursive_schedule",
        name="create_scheduled_work",
        arguments=json.dumps(
            {
                "title": "Recursive task",
                "cron": "0 10 * * 2",
                "prompt": "This must be refused in background work.",
            }
        ),
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(
                    content="Trying to schedule recursively.",
                    tool_calls=(recursive_call,),
                ),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Recursive scheduling was refused."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
        shell=FailingShellBoundary(),
    )
    task = _task()
    state = WorkspaceState(Workspace.from_path(workspace))
    persist_scheduled_work(state.path, (task,))
    scheduled_work_path = state.scheduled_work_path
    persisted_before = scheduled_work_path.read_bytes()

    result = await runtime.scheduled_work_coordinator.trigger(task)

    assert result.status == "completed"
    assert scheduled_work_path.read_bytes() == persisted_before
    session = await runtime.sessions.load(task.session_id)
    recursive_result = session.messages[2]
    assert isinstance(recursive_result, ToolSessionMessage)
    assert recursive_result.name == "create_scheduled_work"
    assert recursive_result.status == "refused"
    assert recursive_result.content == (
        "Scheduled Work creation is unavailable because confirmation is not implemented."
    )


@pytest.mark.asyncio
async def test_runtime_scheduled_work_uses_enabled_web_and_disabled_shell_catalog(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    web_only_config = VALID_CONFIG.replace(
        "[tools.web]\nenabled = false",
        "[tools.web]\nenabled = true",
    ).replace(
        "[tools.shell]\nenabled = true",
        "[tools.shell]\nenabled = false",
    )
    (agent_home / "config.toml").write_text(web_only_config, encoding="utf-8")
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Web catalog checked."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    result = await runtime.scheduled_work_coordinator.trigger(_task())

    assert result.status == "completed"
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert [schema["function"]["name"] for schema in request.tools] == [
        "read_file",
        "list_files",
        "search_files",
        "write_file",
        "edit_file",
        "web_search",
        "web_fetch",
        "create_scheduled_work",
    ]


@pytest.mark.asyncio
async def test_runtime_scheduled_work_calls_registered_workspace_inspection_tools(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    (workspace / "alpha.txt").write_text("needle alpha\n", encoding="utf-8")
    calls = (
        ModelToolCall(id="call_list", name="list_files", arguments="{}"),
        ModelToolCall(
            id="call_search",
            name="search_files",
            arguments='{"query":"needle"}',
        ),
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Inspecting.", tool_calls=calls),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Inspection complete."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    result = await runtime.scheduled_work_coordinator.trigger(_task())

    assert result.status == "completed"
    follow_up = provider.complete_requests[1]
    assert isinstance(follow_up, ModelRequest)
    tool_messages = [
        message for message in follow_up.messages if isinstance(message, ToolModelMessage)
    ]
    assert [(message.name, message.content) for message in tool_messages] == [
        ("list_files", "alpha.txt"),
        ("search_files", "alpha.txt:1:needle alpha"),
    ]
