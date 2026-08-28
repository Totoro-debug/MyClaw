"""Command-line entry point for MyClaw."""

import asyncio
from datetime import datetime
from pathlib import Path
from time import monotonic
from uuid import uuid4

import typer
from rich.console import Console
from tzlocal import get_localzone_name

from myclaw.agent.loop import AgentLoop, SkillContextTooLargeError
from myclaw.agent.message_bus import MessageBus
from myclaw.agent.workspace_state import (
    WorkspaceState,
    WorkspaceStateError,
    normalize_workspace_path,
)
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigError, ConfigLoader, UserConfiguration
from myclaw.errors import ErrorInfo
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import (
    FatalManagementError,
    ManagementError,
    ManagementViewService,
)
from myclaw.memory.dream import Dream
from myclaw.memory.manager import MemoryManager
from myclaw.provider.factory import create_provider
from myclaw.provider.model_router import ModelRouter
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleService
from myclaw.terminal.conversation import (
    TerminalConversationApp,
    is_interactive_terminal,
)
from myclaw.utils.scheduler import AsyncioSchedulerClock

app = typer.Typer(
    add_completion=False,
    help="MyClaw Personal Agent runtime.",
    rich_markup_mode="rich",
)
console = Console()


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _print_error_info(error: ErrorInfo) -> None:
    console.print(
        f"{error.code}: {error.message}",
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


def _print_error(error: ErrorInfo, path: object) -> None:
    _print_error_info(error)
    console.print(f"Path: {path}", markup=False, highlight=False, soft_wrap=True)


async def _run_cli_conversation(
    *,
    agent_home: AgentHome,
    workspace: Path,
    configuration: UserConfiguration,
) -> None:
    """Compose one Runtime Lifetime and run its Terminal Conversation."""
    workspace_state: WorkspaceState | None = None
    router: ModelRouter | None = None
    dream: Dream | None = None
    schedule_service: ScheduleService | None = None
    active_loop: AgentLoop | None = None
    current_loop: AgentLoop | None = None
    bus: MessageBus | None = None
    management: ManagementViewService | None = None
    terminal_app: TerminalConversationApp | None = None
    pending_target: AgentLoop | None = None
    replacement_lock = asyncio.Lock()
    aborted_loops: list[AgentLoop] = []
    closed_loops: list[AgentLoop] = []
    replacement_failed_closed = False
    started = False
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []

    async def abort_loop_once(loop: AgentLoop) -> None:
        if any(loop is existing for existing in aborted_loops):
            return
        aborted_loops.append(loop)
        await loop.abort()

    async def close_loop_once(loop: AgentLoop) -> None:
        if any(loop is existing for existing in closed_loops):
            return
        closed_loops.append(loop)
        await loop.close()

    async def abort_target_for_management(target: AgentLoop) -> None:
        try:
            await abort_loop_once(target)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise ManagementError(
                ErrorInfo(
                    "persistence_error",
                    "Conversation Session could not be prepared.",
                )
            ) from error

    try:
        workspace_path = normalize_workspace_path(workspace)
        workspace_state = WorkspaceState(workspace_path)
        workspace_state.initialize(agent_home_root=agent_home.path)

        bus = MessageBus()
        router = ModelRouter(
            configuration=configuration,
            provider_factory=create_provider,
        )
        memory_manager = MemoryManager(workspace_state)
        dream = Dream(
            memory_manager=memory_manager,
            model_router=router,
            batch_size=configuration.memory.batch_size,
            max_iterations=configuration.runtime.max_iterations,
        )

        async def execute_user_job(job: ScheduleJob) -> None:
            if current_loop is None:
                raise RuntimeError("Schedule Service user executor is not bound")
            await current_loop.run_schedule_job(job)

        async def wait_for_session_persist(loop: AgentLoop, session_id: str) -> None:
            old_session = loop.session
            if old_session.session_id != session_id:
                return
            try:
                await old_session.wait_for_pending_persist()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise ManagementError(
                    ErrorInfo(
                        "persistence_error",
                        "Conversation Session could not be prepared.",
                    )
                ) from error

        async def prepare_session_resume(session_id: str) -> None:
            old_loop = current_loop
            if old_loop is None:
                raise ManagementError(
                    ErrorInfo("route_unavailable", "Runtime Generation is unavailable.")
                )
            await wait_for_session_persist(old_loop, session_id)

        schedule_service = ScheduleService(
            workspace_state=workspace_state,
            clock=AsyncioSchedulerClock(now=_local_now),
            execute_user_job=execute_user_job,
            execute_dream=dream.run,
            timezone_name=get_localzone_name(),
        )

        def create_agent_loop(session_id: str | None) -> AgentLoop:
            return AgentLoop(
                workspace_path=workspace_path,
                workspace_state=workspace_state,
                agent_home=agent_home,
                configuration=configuration,
                bus=bus,
                schedule_service=schedule_service,
                model_router=router,
                memory_manager=memory_manager,
                session_id=session_id,
                now=_local_now,
                new_uuid=uuid4,
                monotonic_now=monotonic,
            )

        def current_agent_loop() -> AgentLoop:
            if current_loop is None:
                raise ManagementError(
                    ErrorInfo("route_unavailable", "Runtime Generation is unavailable.")
                )
            return current_loop

        async def replace_agent_loop(session_id: str, force: bool) -> None:
            nonlocal active_loop, current_loop, pending_target, replacement_failed_closed
            async with replacement_lock:
                old_loop = current_loop
                if old_loop is None:
                    raise ManagementError(
                        ErrorInfo("route_unavailable", "Runtime Generation is unavailable.")
                    )

                target: AgentLoop | None = None
                replacement_barrier_held = False
                destructive_started = False

                async def release_replacement_barrier(*, resume_inbound: bool) -> None:
                    nonlocal replacement_barrier_held
                    if not replacement_barrier_held:
                        return
                    await old_loop._release_replacement_barrier(
                        resume_inbound=resume_inbound
                    )
                    replacement_barrier_held = False

                async def reject_prepared_target(target: AgentLoop) -> None:
                    nonlocal pending_target
                    try:
                        await abort_target_for_management(target)
                    finally:
                        pending_target = None
                        await release_replacement_barrier(resume_inbound=True)

                try:
                    await old_loop._pause_for_replacement()
                    replacement_barrier_held = True
                    await wait_for_session_persist(old_loop, session_id)

                    target = create_agent_loop(session_id)
                    pending_target = target
                    target.preflight()
                except asyncio.CancelledError as cancellation:
                    cleanup_error: BaseException | None = None
                    if target is not None:
                        try:
                            await abort_loop_once(target)
                        except BaseException as caught:
                            cleanup_error = caught
                    pending_target = None
                    await release_replacement_barrier(resume_inbound=True)
                    if cleanup_error is not None:
                        raise cancellation from cleanup_error
                    raise
                except Exception as error:
                    target_cleanup_error: Exception | None = None
                    if target is not None:
                        try:
                            await abort_target_for_management(target)
                        except asyncio.CancelledError:
                            raise
                        except Exception as caught:
                            target_cleanup_error = caught
                    pending_target = None
                    await release_replacement_barrier(resume_inbound=True)
                    if target_cleanup_error is not None:
                        raise ManagementError(
                            ErrorInfo(
                                "persistence_error",
                                "Conversation Session could not be prepared.",
                            )
                        ) from BaseExceptionGroup(
                            "Conversation Session preparation cleanup failed",
                            (error, target_cleanup_error),
                        )
                    raise ManagementError(
                        ErrorInfo(
                            "persistence_error",
                            "Conversation Session could not be prepared.",
                        )
                    ) from error

                assert target is not None
                try:
                    has_active_run = old_loop.control.has_active_run
                except Exception as error:
                    await reject_prepared_target(target)
                    raise ManagementError(
                        ErrorInfo("route_unavailable", "Runtime Generation is unavailable.")
                    ) from error
                if has_active_run and not force:
                    await reject_prepared_target(target)
                    raise ManagementError(
                        ErrorInfo(
                            "model_invalid_request",
                            "An active foreground run must be confirmed before switching Sessions.",
                        )
                    )

                if terminal_app is None or schedule_service is None or bus is None:
                    await reject_prepared_target(target)
                    raise ManagementError(
                        ErrorInfo("route_unavailable", "Session resume is unavailable.")
                    )

                destructive_started = True
                try:
                    await terminal_app.quiesce_for_rebind()
                    await schedule_service.pause_and_drain()
                    current_loop = None
                    await abort_loop_once(old_loop)
                    await bus.reset()
                    await terminal_app.rebind_agent_loop(
                        control=target.control,
                        skill_metadata=target.skill_metadata,
                        session_projection=target.project_foreground_conversation(),
                    )
                    await target.start()
                    current_loop = target
                    active_loop = target
                    pending_target = None
                    await release_replacement_barrier(resume_inbound=True)
                    schedule_service.resume()
                except asyncio.CancelledError:
                    replacement_failed_closed = True
                    current_loop = None
                    if management is not None:
                        management.deactivate()
                    raise
                except BaseException as error:
                    replacement_failed_closed = True
                    current_loop = None
                    if management is not None:
                        management.deactivate()
                    raise FatalManagementError(
                        ErrorInfo(
                            "persistence_error",
                            "Runtime Session replacement could not be completed.",
                        )
                    ) from error
                finally:
                    await release_replacement_barrier(
                        resume_inbound=not destructive_started
                    )

        initial_loop = create_agent_loop(None)
        active_loop = initial_loop
        initial_loop.preflight()
        schedule_service._prepare_start()
        await schedule_service.register_dream_job(
            schedule=JobSchedule.from_cron_input(
                configuration.memory.schedule,
                get_localzone_name(),
            )
        )
        current_loop = initial_loop

        management = ManagementViewService(
            agent_home,
            current_agent_loop=current_agent_loop,
            workspace_state=workspace_state,
            replace_agent_loop=replace_agent_loop,
            prepare_session_resume=prepare_session_resume,
            now=_local_now,
            monotonic=monotonic,
            current_memory_manager=lambda: memory_manager,
            current_dream=lambda: dream,
            schedule_status=lambda: schedule_service.status_snapshot().to_dict(),
        )
        dispatcher = ManagementCommandDispatcher(management)
        terminal_app = TerminalConversationApp(
            bus=bus,
            control=initial_loop.control,
            management_dispatcher=dispatcher,
            skill_metadata=initial_loop.skill_metadata,
        )

        await initial_loop.start()
        schedule_service.start()
        started = True
        await terminal_app.run_async()
        fatal_management_error = getattr(terminal_app, "fatal_management_error", None)
        if isinstance(fatal_management_error, FatalManagementError):
            raise fatal_management_error
    except BaseException as error:
        primary_error = error
    finally:
        if management is not None:
            try:
                management.deactivate()
            except BaseException as error:
                cleanup_errors.append(error)

        if schedule_service is not None:
            try:
                await schedule_service.pause_and_drain()
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                await schedule_service.close()
            except BaseException as error:
                cleanup_errors.append(error)

        if pending_target is not None:
            try:
                await abort_loop_once(pending_target)
            except BaseException as error:
                cleanup_errors.append(error)

        if active_loop is not None:
            try:
                if started and active_loop is current_loop and not replacement_failed_closed:
                    await close_loop_once(active_loop)
                else:
                    await abort_loop_once(active_loop)
            except BaseException as error:
                cleanup_errors.append(error)

        if dream is not None:
            try:
                await dream.close()
            except BaseException as error:
                cleanup_errors.append(error)

        if router is not None:
            try:
                await router.close()
            except BaseException as error:
                cleanup_errors.append(error)

    if primary_error is not None:
        if cleanup_errors:
            cleanup = (
                cleanup_errors[0]
                if len(cleanup_errors) == 1
                else BaseExceptionGroup("CLI shutdown failed", cleanup_errors)
            )
            raise primary_error from cleanup
        raise primary_error
    if cleanup_errors:
        cleanup = (
            cleanup_errors[0]
            if len(cleanup_errors) == 1
            else BaseExceptionGroup("CLI shutdown failed", cleanup_errors)
        )
        raise cleanup


@app.callback(invoke_without_command=True)
def main(context: typer.Context) -> None:
    """Start the MyClaw Personal Agent."""
    if context.invoked_subcommand is not None:
        return
    agent_home = AgentHome.production()
    loader = ConfigLoader(agent_home)
    try:
        configuration = loader.load_for_startup()
    except ConfigError as config_error:
        _print_error(config_error.error, loader.path)
        exit_code = 1 if config_error.error.code == "persistence_error" else 2
        raise typer.Exit(code=exit_code) from None
    except OSError:
        _print_error(
            ErrorInfo("persistence_error", "User Configuration could not be read or written."),
            loader.path,
        )
        raise typer.Exit(code=1) from None
    if not is_interactive_terminal():
        _print_error_info(
            ErrorInfo(
                "interactive_terminal_required",
                "Terminal Conversation requires interactive stdin, stdout, and stderr TTYs.",
            )
        )
        raise typer.Exit(code=2)
    try:
        asyncio.run(
            _run_cli_conversation(
                agent_home=loader.agent_home,
                workspace=Path.cwd(),
                configuration=configuration,
            )
        )
    except WorkspaceStateError as workspace_state_error:
        _print_error_info(workspace_state_error.error)
        raise typer.Exit(code=1) from None
    except SkillContextTooLargeError as skill_error:
        _print_error_info(skill_error.error)
        raise typer.Exit(code=1) from None
    except FatalManagementError as fatal_error:
        _print_error_info(fatal_error.error)
        raise typer.Exit(code=1) from None
    except Exception as startup_error:
        error_info = getattr(startup_error, "error", None)
        if not isinstance(error_info, ErrorInfo):
            error_info = ErrorInfo(
                "persistence_error",
                "MyClaw runtime could not be started.",
            )
        _print_error_info(error_info)
        raise typer.Exit(code=1) from None


@app.command("config")
def config_command() -> None:
    """Display User Configuration with plaintext API keys redacted."""
    agent_home = AgentHome.production()
    loader = ConfigLoader(agent_home)
    try:
        loader.ensure_default()
        view = loader.view()
    except (OSError, UnicodeError):
        _print_error(
            ErrorInfo("persistence_error", "User Configuration could not be read or written."),
            loader.path,
        )
        raise typer.Exit(code=1) from None

    if view.error is not None:
        _print_error(view.error, view.path)
    else:
        console.print(
            f"Path: {view.path}",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
    console.print(
        view.redacted_content,
        markup=False,
        highlight=False,
        soft_wrap=True,
        end="" if view.redacted_content.endswith("\n") else "\n",
    )
    if view.error is not None:
        raise typer.Exit(code=2)
