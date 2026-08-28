# Issue #201 Test Migration Ledger

This ledger records the test-surface migration made while removing the legacy
`myclaw.agent.runtime` aggregate. The baseline is the clean collection at
`27f6550cfc4a2c074a8ecc7bf5a5a7d0d3768c25`.

## Collection accounting

| Measure | Nodes |
| --- | ---: |
| Baseline collection | 1553 |
| Final collection | 1439 |
| Deleted aggregate-file nodes | 96 |
| Deleted Terminal RuntimeHost-only nodes | 32 |
| Deleted `_run_runtime_conversation` FakeRuntime nodes | 7 |
| Removed legacy side of the parameterized replacement contract | 1 |
| Removed obsolete session-authority bypass node | 1 |
| Added public-seam migration contracts | 21 |
| Added source/import and clean-distribution absence contracts | 2 |
| Net change | -114 |

The 137 deleted nodes belonged to the legacy aggregate, adapter, or
RuntimeHost fixture. Twenty-one consolidated contracts were added at the current
CLI, AgentLoop, and Terminal public seams where the old nodes contained
observable assertions that were not already proved there; two more contracts
prove source/import and built-distribution absence. The remaining reduction is
legacy-owner parameter expansion or white-box lifecycle coverage whose public
assertions are retained by the targets below. Parameterized function names are
listed once; the collection accounting includes every expanded pytest node.

## Deleted aggregate nodes

Each exact node listed in a block maps to the public target and assertion
classes stated immediately after that block.

### `tests/test_runtime.py` - composition, context, and startup

```text
test_runtime_prepare_does_not_accept_generation_collaborators
test_runtime_preserves_explicit_foreground_timezone
test_runtime_composes_context_builder_only_inside_agent_loop
test_runtime_persists_the_configured_dream_schedule_with_startup_timezone
test_injected_skill_catalog_root_is_the_only_confirmation_free_skill_root
test_foreground_skill_catalog_is_included_in_the_exact_budget_guard
test_always_skill_budget_allows_exact_foreground_projection
test_always_skill_budget_overflow_fails_before_provider_or_background_tasks
test_always_skill_budget_ignores_retained_session_history
test_metadata_only_skill_skips_startup_skill_budget_preflight
test_conversation_summary_provider_keeps_skill_metadata_out_of_its_prompt
test_foreground_context_uses_one_staged_blackboard_for_summary_and_chat_projection
test_manual_body_counts_in_foreground_token_budget_but_not_summary_provider
test_oversized_manual_body_returns_context_overflow_without_provider_or_commit
test_management_command_performs_zero_task_framing_attempts
test_runtime_composition_rejects_invalid_discovered_iana_before_provider_call
test_runtime_leaves_legacy_schedule_state_untouched
test_runtime_ignores_legacy_schedule_state_path_types
```

Assertion classes: architecture, startup, identity, security, failure, and
persist. Targets: `tests/test_cli.py` composition/startup and Skill error
contracts; `tests/agent/test_loop.py` composition, framing, projection, and
failure contracts; `tests/agent/test_context.py` context and Skill metadata;
`tests/memory/test_conversation_summary.py` summary budget/projection; and
`tests/scheduling/test_schedule_service.py` schedule clock/state boundaries.

### `tests/test_runtime.py` - Session, logs, and provider route behavior

```text
test_prepared_runtime_correlates_foreground_and_title_work_with_its_session
test_concurrent_foreground_sessions_write_only_to_their_own_session_logs
test_unavailable_session_log_preserves_events_session_and_tool_failure
test_foreground_tool_diagnostics_preserve_boundary_exception_details
test_foreground_model_failure_keeps_event_safe_without_log_redaction
test_prepared_repl_defers_injected_provider_factory_until_first_nonblank_input
test_prepared_repl_uses_the_chat_model_route
test_default_task_framer_uses_model_router_retry_before_one_foreground_run
test_default_task_framer_uses_model_router_default_fallback
test_prepared_repl_routes_transient_provider_failures_through_one_retry_budget
test_prepared_repl_status_reports_the_actual_fallback_route_and_session
test_runtime_status_estimate_omits_a_pure_error_assistant
test_runtime_status_estimate_includes_the_interrupted_history_marker
test_prepared_repl_defers_an_unusable_default_until_route_use
test_prepared_repl_uses_the_effective_fallback_route_budget
test_prepared_repl_reuses_one_session_and_its_startup_system_context
```

Assertion classes: startup, failure, order, persist, and security. Exact
migration targets include
`tests/agent/test_fixed_catalog.py::test_agent_loop_continues_when_session_log_path_is_unavailable`,
`test_agent_loop_tool_failure_keeps_private_diagnostics_out_of_public_output`,
and `test_agent_loop_model_failure_logs_private_cause_but_emits_safe_terminal`.
The remaining route/retry/status assertions are owned by
`tests/agent/test_loop.py`, `tests/agent/test_runner.py`,
`tests/test_model_router.py`, and
`tests/management/test_management_views.py`. `RuntimeStatus` is a retained
Management domain model, not a legacy aggregate.

### `tests/test_runtime_active_session.py`

```text
test_runtime_routes_turn_title_status_and_close_through_one_active_session
test_late_title_is_saved_by_the_next_complete_turn
test_runtime_rejects_tool_call_title_and_counts_its_usage
test_runtime_empty_normalized_title_uses_the_first_user_fallback
test_runtime_shutdown_applies_first_user_title_fallback_before_final_save
test_immediate_turn_cancellation_keeps_the_first_user_title_lifecycle
test_runtime_shutdown_keeps_an_empty_session_memory_only
test_runtime_shutdown_swallows_final_session_close_fault_before_router_close
test_runtime_replacement_abandons_old_and_normal_close_saves_target
test_forced_runtime_replacement_cancels_blocked_framing_without_late_session_writes
test_runtime_active_session_keeps_artifact_and_log_correlation_when_persist_fails
test_runtime_summary_uses_the_effective_route_and_advances_the_active_session
```

Assertion classes: identity, order, startup, cancellation, persist, failure,
and replacement. Title and final-save semantics map exactly to
`tests/agent/test_loop.py::test_late_title_is_persisted_by_the_next_completed_turn`,
`test_invalid_title_uses_first_input_fallback_and_keeps_usage`,
`test_close_applies_first_input_title_fallback_before_final_save`, and
`test_loop_close_swallows_final_session_failure`. Artifact/persist correlation
maps to
`tests/agent/test_fixed_catalog.py::test_agent_loop_keeps_artifact_and_log_correlation_when_persist_fails`.
Session snapshot/abandon/close ownership remains in
`tests/sessions/test_session.py`; forced and same-Session replacement remain in
`tests/test_cli.py` and
`tests/test_cli_replacement_contract.py::test_cli_force_replacement_cancels_framing_without_old_session_late_writes`.

### `tests/test_runtime_generation.py` - management and replacement

```text
test_management_port_identity_survives_runtime_generation_replacement
test_management_uses_target_generation_resources_after_replacement
test_management_is_unavailable_between_generation_detach_and_start
test_runtime_host_refreshes_skill_snapshot_across_generation_replacement
test_runtime_rejects_conflicting_dream_state_before_agent_loop_construction
test_runtime_host_refreshes_skill_snapshot_after_generation_skill_deletion
test_runtime_host_refreshes_skill_snapshot_when_resuming_current_session
test_runtime_host_refreshes_frozen_always_body_across_generation_replacement
test_target_generation_preparation_failure_preserves_old_generation
test_target_schedule_service_preflight_failure_preserves_the_started_old_generation
test_abort_interrupts_an_in_progress_normal_close_before_final_session_save
test_normal_close_waits_for_an_in_progress_dream_before_closing_the_router
test_generation_replacement_aborts_old_dream_and_schedule_service
test_runtime_abort_drains_active_dream_and_schedule_service
test_runtime_abort_drains_only_inbound_after_unbinding_its_callback
test_generation_replacement_finishes_atomically_when_waiter_is_cancelled
test_runtime_composition_failure_closes_a_constructed_dream
test_pending_only_resume_replaces_every_generation_owned_component
test_terminal_rebinds_once_and_rebuilds_the_target_session
test_runtime_handoff_preserves_lifetime_seams_and_starts_target_after_rebind
test_runtime_handoff_waits_for_old_tasks_before_reset_rebind_and_target_output
test_target_preflight_failure_preserves_current_generation_and_bus
test_rebind_failure_leaves_a_quiesced_target_without_mixed_generation_state
test_target_start_failure_enters_the_same_fail_closed_state
test_concurrent_old_generation_resume_requests_cannot_replace_the_new_generation
test_management_command_uses_one_generation_port_across_concurrent_rebind
test_close_waits_for_an_in_progress_replacement_and_closes_the_committed_target
test_active_same_session_resume_requires_confirmation_before_rebuild
test_active_resume_decline_keeps_the_old_generation_untouched
test_active_resume_approval_detaches_without_waiting_for_provider_close
```

Assertion classes: identity, replacement, startup, failure, cancellation,
order, security, and persist. Targets: the CLI public replacement tests
`test_cli_resume_preflight_failure_preserves_current_generation`,
`test_cli_resume_publishes_current_only_after_target_activation`,
`test_cli_same_session_resume_waits_for_pending_persist_before_target_load`,
`test_cli_resume_destructive_failure_fails_closed_and_aborts_each_loop_once`,
and `test_cli_reports_fatal_replacement_failure_once_without_raw_exception_output`;
the AgentLoop replacement barrier/abort/close tests; the Management current
provider/unavailable-window tests; `test_cli_replacement_contract.py` for
same-Session Skill changed/deleted/always-body freshness through both Terminal
metadata and the replacement foreground system prompt, Bus identity,
Management unavailability, pre-start target isolation, projection, and event
order; and
`tests/terminal/test_conversation.py::test_active_resume_decline_then_force_rebinds_the_same_bus`
for the user-visible decline/force flow.

### `tests/test_runtime_shutdown.py`

```text
test_runtime_shutdown_cancels_blocked_framing_and_reclaims_title_work
test_scheduler_preflight_failure_starts_no_runtime_tasks
test_start_preflight_failure_requires_no_async_cleanup
test_runtime_start_activation_failure_aborts_all_owned_tasks
test_normal_repl_exit_closes_the_runtime_model_provider
test_normal_eof_and_exit_shutdown_do_not_create_diagnostic_log
test_prepared_runtime_rejects_a_second_repl_invocation
test_runtime_run_preserves_the_primary_error_when_cleanup_also_fails
test_external_runtime_close_waits_for_the_repl_and_input_to_stop
test_writer_failure_finishes_runtime_shutdown_without_task_leaks
test_repeated_and_idle_cancellations_cancel_only_foreground_until_exit
```

Assertion classes: startup, cancellation, failure, persist, order, and
security. Targets: CLI initial-startup/finally-close/error-redaction tests;
AgentLoop activation, abort, close, cancellation, and title-work tests; and
Terminal direct-loop cleanup tests.

### `tests/test_runtime_session_title.py`

```text
test_title_finishes_before_chat_when_direct_provider_exposes_route_status
test_prepared_runtime_uses_an_isolated_chat_stream_for_session_title
test_existing_session_turn_does_not_regenerate_its_title
```

Assertion classes: order, identity, and persist. Targets:
`tests/agent/test_loop.py::test_first_message_title_runs_while_foreground_chat_is_blocked`,
`tests/agent/test_loop.py::test_title_usage_is_preserved_when_title_finishes_before_foreground_commit`,
`tests/agent/test_loop.py::test_slow_title_keeps_one_session_log_owner_across_the_next_fifo_turn`,
and Session metadata tests.

## Deleted CLI adapter nodes

The following seven collected nodes exercised only the removed
`cli._run_runtime_conversation(FakeRuntime)` adapter. Their observable
startup, mount, run, cancellation, queue, and close/error behavior is covered
by the current `_run_cli_conversation` tests and the direct Terminal lifecycle
tests:

```text
test_composition_driver_owns_runtime_lifecycle_outside_terminal_mount
test_composition_driver_closes_runtime_when_initial_start_fails
test_composition_driver_closes_runtime_when_setup_fails[app_init]
test_composition_driver_closes_runtime_when_setup_fails[bind]
test_composition_driver_closes_once_when_application_is_cancelled
test_composition_driver_preserves_messages_queued_before_app_mount
test_composition_driver_retrieves_close_task_that_emits_after_app_exit
```

Targets: `tests/test_cli.py::test_cli_async_root_owns_lifetime_components_and_async_shutdown`,
`test_cli_async_root_cleans_partial_startup_without_registering_dream_job`,
`test_cli_reports_unexpected_startup_failure_without_raw_exception_output`,
`test_cli_reports_fatal_replacement_failure_once_without_raw_exception_output`,
and `tests/terminal/test_conversation.py` direct-loop cleanup/cancellation
tests. The removed nodes contained no product behavior not exercised by those
public seams.

## Deleted Terminal RuntimeHost-only nodes

These nodes depended on the removed `_GenerationHost`/RuntimeHost fixture. The
rebind and management assertions were retained in direct Terminal/CLI tests;
session listing and persistence assertions remain in the Session and CLI
public seams. The names below are the exact removed function nodes, including
the two parameterized families in their collection accounting.

```text
test_runtime_rebind_clears_stale_skill_completion_state
test_supported_management_commands_use_the_prepared_runtime_without_session_messages
test_inexact_slash_input_reaches_the_active_message_bus_unchanged
test_empty_resume_picker_cancellation_preserves_existing_management_rows
test_resume_opens_a_picker_with_title_and_local_update_time
test_resume_selection_rebuilds_the_display_from_the_selected_session
test_resume_rebuilds_a_successful_tool_run_activity_group
test_resume_groups_a_recognizable_tool_result_after_the_final_response
test_resume_uses_the_last_completed_no_tool_assistant_as_final_response
test_resume_expands_cancelled_and_failed_activity_groups
test_resume_keeps_unknown_outcome_expanded_without_inventing_status
test_resume_clamps_reversed_historical_duration_to_zero
test_resume_keeps_pre_user_messages_and_unclassifiable_runs_flat
test_resume_picker_cancellation_and_outside_click_preserve_current_display
test_resume_picker_mouse_selection_switches_to_the_clicked_session
test_resume_failure_after_a_stale_listing_preserves_the_current_display
test_resume_picker_scrolls_in_management_order_and_selects_by_keyboard
test_resume_picker_reports_corrupt_entries_without_mutating_them
test_resume_listing_failure_preserves_session_and_existing_display
test_resume_requires_result_and_runtime_authority_to_agree
test_unexpected_resume_exception_preserves_session_display_and_interaction
test_fatal_resume_failure_exits_terminal_without_rendering_raw_error
test_resume_selection_serializes_input_until_rebuild_finishes
test_resumed_long_history_starts_latest_and_preserves_runtime_input_history
```

Assertion classes: Terminal presentation/order, management identity and
unavailable failure, Session persist/security, cancellation, and replacement.
The consolidated Terminal migrations are
`test_resume_picker_orders_sessions_and_cancellation_preserves_display`,
`test_resume_selection_rebinds_sanitized_session_projection`,
`test_active_resume_decline_then_force_rebinds_the_same_bus`,
`test_resume_picker_mouse_selection_rebinds_the_clicked_session`,
`test_resume_picker_scrolls_in_management_order_and_selects_by_keyboard`,
`test_resume_serializes_input_until_cli_rebind_finishes`,
`test_resumed_long_history_starts_latest_and_preserves_input_history`,
`test_resume_projects_unknown_reversed_and_unclassifiable_history_safely`,
`test_resume_stale_selection_preserves_current_display_and_interaction`, and
`test_fatal_resume_failure_exits_without_rendering_private_error` in
`tests/terminal/test_conversation.py`. They exercise the real CLI composition
callback and current Terminal rebind surface, including old-generation stream
cleanup before display rebuild. Detailed Session listing,
corrupt-entry, and ordering rules remain in
`tests/sessions/test_session_resume.py`; current-provider and unavailable-window
rules remain in `tests/management/test_management_views.py`.

## Rewritten or renamed nodes

| Legacy node | Current node | Assertion classes and owner |
| --- | --- | --- |
| `tests/agent/test_fixed_catalog_runtime.py::test_runtime_uses_fixed_catalog_for_provider_confirmation_and_persistence` | `tests/agent/test_fixed_catalog.py::test_agent_loop_uses_fixed_catalog_for_provider_confirmation_and_persistence` | security, persist; AgentLoop/ToolGateway |
| `tests/agent/test_fixed_catalog_runtime.py::test_runtime_reads_known_skill_path_without_confirmation` | `tests/agent/test_fixed_catalog.py::test_agent_loop_reads_known_skill_path_without_confirmation` | security; AgentLoop/Skill catalog |
| `tests/agent/test_fixed_catalog_runtime.py::test_runtime_advertises_and_persists_multiple_autonomous_skill_reads` | `tests/agent/test_fixed_catalog.py::test_agent_loop_advertises_and_persists_multiple_autonomous_skill_reads` | persist, order; AgentLoop/Session |
| `tests/agent/test_fixed_catalog_runtime.py::test_runtime_cancellation_reaches_an_active_fixed_catalog_tool` | `tests/agent/test_fixed_catalog.py::test_agent_loop_cancellation_reaches_an_active_fixed_catalog_tool` | cancellation; AgentLoop |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_conversation_manages_schedule_jobs_without_confirmation` | `tests/scheduling/test_schedule_agent_loop.py::test_agent_loop_manages_schedule_jobs_without_confirmation` | security, identity; AgentLoop/Schedule |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_schedule_uses_its_own_complete_context_projection` | `tests/scheduling/test_schedule_agent_loop.py::test_schedule_uses_its_own_complete_context_projection` | identity, order; Schedule/AgentLoop |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_schedule_tool_loop_persists_each_message_from_awaitable_run` | `tests/scheduling/test_schedule_agent_loop.py::test_schedule_tool_loop_persists_each_message_from_awaitable_run` | persist, order; Schedule/Session |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_schedule_tool_loop_does_not_prepare_summary_inside_agent_run` | `tests/scheduling/test_schedule_agent_loop.py::test_schedule_tool_loop_does_not_prepare_summary_inside_agent_run` | order, identity; Schedule/Memory |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_schedule_summary_flows_through_memory_to_a_later_schedule_run` | `tests/scheduling/test_schedule_agent_loop.py::test_schedule_summary_flows_through_memory_to_a_later_schedule_run` | persist, order; Schedule/Memory |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_dispatcher_wakes_for_due_at_job_and_keeps_schedule_session_out_of_resume` | `tests/scheduling/test_schedule_agent_loop.py::test_schedule_dispatcher_wakes_for_due_at_job_and_keeps_schedule_session_out_of_resume` | startup, identity; Schedule/Session |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_wires_schedule_service_user_executor_to_agent_loop` | `tests/scheduling/test_schedule_agent_loop.py::test_schedule_service_user_executor_is_bound_to_agent_loop` | identity; Schedule/AgentLoop |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_foreground_and_schedule_share_runner_and_gateway_identity` | `tests/scheduling/test_schedule_agent_loop.py::test_foreground_and_schedule_share_runner_and_gateway_identity` | identity; AgentLoop/Schedule |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_externalizers_keep_foreground_and_schedule_artifacts_separate` | `tests/scheduling/test_schedule_agent_loop.py::test_foreground_and_schedule_artifacts_remain_separate` | security, identity; Session/Schedule |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_schedule_session_uses_schedule_clock_for_persisted_timestamps` | `tests/scheduling/test_schedule_agent_loop.py::test_schedule_session_uses_schedule_clock_for_persisted_timestamps` | persist; Schedule/Session |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_shutdown_during_schedule_model_persists_user_and_keeps_job_pending` | `tests/scheduling/test_schedule_agent_loop.py::test_schedule_shutdown_during_model_persists_user_and_keeps_job_pending` | cancellation, persist; Schedule/Session |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_shutdown_during_schedule_preparation_persists_user` | `tests/scheduling/test_schedule_agent_loop.py::test_schedule_shutdown_during_preparation_persists_user` | cancellation, persist; Schedule/Session |
| `tests/scheduling/test_schedule_runtime.py::test_runtime_schedule_failure_logs_one_safe_session_warning` | `tests/scheduling/test_schedule_agent_loop.py::test_schedule_failure_logs_one_safe_session_warning` | failure, security; Schedule/Session Log |
| `tests/scheduling/test_schedule_service.py::test_prepared_runtime_executes_at_job_with_schedule_route_and_partition` | `tests/scheduling/test_schedule_service.py::test_agent_loop_executes_at_job_with_schedule_route_and_partition` | identity, order; Schedule/AgentLoop |
| `tests/scheduling/test_schedule_service.py::test_prepared_runtime_runs_foreground_while_every_job_is_active` | `tests/scheduling/test_schedule_service.py::test_agent_loop_runs_foreground_while_every_job_is_active` | order, cancellation; Schedule/AgentLoop |
| `tests/memory/test_memory_task.py::test_runtime_dream_uses_memory_route_with_static_default_fallback` | `tests/memory/test_dream.py::test_dream_uses_the_memory_route_with_static_default_fallback` | identity, failure; Memory/Dream |
| `tests/sessions/test_session_resume.py::test_resume_selects_the_loaded_session_for_the_runtime_owner` | `tests/sessions/test_session_resume.py::test_resume_selects_the_loaded_session_for_the_agent_loop_owner` | identity, persist; Session/AgentLoop |
| `tests/terminal/test_conversation.py::test_terminal_conversation_uses_the_prepared_runtime_lifecycle` | `tests/terminal/test_conversation.py::test_terminal_conversation_uses_the_direct_terminal_loop_lifecycle` | startup, order, failure; Terminal/AgentLoop |
| `tests/terminal/test_conversation.py::test_prepared_runtime_exec_confirmation_preserves_the_exact_long_command` | `tests/terminal/test_conversation.py::test_direct_terminal_loop_exec_confirmation_preserves_the_exact_long_command` | security; Terminal/AgentLoop |
| `tests/terminal/test_conversation.py::test_prepared_runtime_close_cancels_the_pending_confirmation_future` | `tests/terminal/test_conversation.py::test_direct_terminal_loop_close_cancels_the_pending_confirmation_future` | cancellation; Terminal/AgentLoop |
| `tests/terminal/test_conversation.py::test_prepared_runtime_cancellation_preserves_partial_and_allows_next_turn` | `tests/terminal/test_conversation.py::test_direct_terminal_loop_cancellation_preserves_partial_and_allows_next_turn` | cancellation, order; Terminal/AgentLoop |
| `tests/terminal/test_conversation.py::test_runtime_cleanup_failure_does_not_mask_an_application_failure` | `tests/terminal/test_conversation.py::test_terminal_cleanup_failure_does_not_mask_an_application_failure` | failure, order; Terminal |
| `tests/terminal/test_conversation.py::test_runtime_start_failure_closes_after_external_driver_start` | `tests/terminal/test_conversation.py::test_terminal_start_failure_closes_after_external_driver_start` | startup, failure; Terminal |
| `tests/terminal/test_conversation.py::test_runtime_cleanup_failure_still_restores_terminal_first` | `tests/terminal/test_conversation.py::test_terminal_cleanup_failure_still_restores_terminal_first` | failure, order; Terminal |
| `tests/terminal/test_conversation.py::test_runtime_start_and_cleanup_failure_preserves_the_start_error` | `tests/terminal/test_conversation.py::test_terminal_start_and_cleanup_failure_preserves_the_start_error` | startup, failure; Terminal |
| `tests/test_runtime_replacement_contract.py::test_cli_and_legacy_share_same_session_replacement_release_contract[cli]` | `tests/test_cli_replacement_contract.py::test_cli_same_session_replacement_keeps_public_generation_contract` | replacement, identity, order, persist; CLI |
| `tests/test_runtime_replacement_contract.py::test_cli_and_legacy_share_same_session_replacement_release_contract[legacy]` | deleted as the legacy-owner duplicate; the CLI node above retains the public contract | replacement, identity, order; CLI |
| `tests/test_permission_loop.py::test_foreground_mutations_execute_without_a_permission_pause` | same node, direct `AgentLoop`/`MessageBus` composition | security, identity; AgentLoop/ToolGateway |

## Standards 2 and 3 convergence addendum

This addendum records the final-interface cleanup performed from clean collection
`dc3ff20f4a9ac290f4b3b57934c102356b02b929`. Parameterized nodes are listed
individually because every expanded node was part of the measured baseline.

### Collection accounting

| Measure | Nodes |
| --- | ---: |
| Convergence baseline collection | 1448 |
| Deleted `test_memory_task.py` nodes | -30 |
| Deleted Memory facade-only node | -1 |
| Deleted REPL-file nodes | -10 |
| Deleted headless Terminal wrapper nodes | -6 |
| Added `MemoryManager` final-interface contracts | +6 |
| Added `Dream` final-interface contracts | +5 |
| Added Management result-projection contracts | +3 |
| Added Terminal unknown-slash contract | +1 |
| Added source/import/distribution absence contract | +1 |
| Final collection | 1417 |

The equation balances exactly: `1448 - 30 - 1 - 10 - 6 + 6 + 5 + 3 + 1 + 1 = 1417`.

### Deleted Memory nodes

| Deleted node | Surviving target or disposition |
| --- | --- |
| `tests/memory/test_memory_task.py::test_summary_store_returns_the_limited_batch_after_the_cursor` | `tests/memory/test_memory_manager.py::test_manager_appends_and_claims_summaries_with_cursor_preadvance` |
| `tests/memory/test_memory_task.py::test_memory_store_treats_a_missing_summary_cursor_as_zero` | `tests/memory/test_memory_manager.py::test_manager_appends_and_claims_summaries_with_cursor_preadvance` |
| `tests/memory/test_memory_task.py::test_memory_store_atomically_persists_the_canonical_summary_cursor` | `tests/memory/test_memory_manager.py::test_manager_appends_and_claims_summaries_with_cursor_preadvance` |
| `tests/memory/test_memory_task.py::test_memory_store_atomically_replaces_exact_long_term_memory` | `tests/memory/test_memory_manager.py::test_manager_reads_disk_and_refreshes_snapshot_after_an_edit` |
| `tests/memory/test_memory_task.py::test_memory_tools_export_common_schemas_with_zero_retries` | `tests/memory/test_dream.py::test_dream_processes_claimed_summaries_through_restricted_memory_route` |
| `tests/memory/test_memory_task.py::test_manual_memory_task_returns_exact_zero_work_result_without_a_model_call` | `tests/memory/test_dream.py::test_dream_returns_without_a_provider_call_when_no_summary_is_pending` |
| `tests/memory/test_memory_task.py::test_wait_until_idle_returns_for_idle_and_current_memory_tasks` | `tests/memory/test_dream.py::test_dream_close_waits_for_active_work_and_releases_the_task` |
| `tests/memory/test_memory_task.py::test_memory_task_uses_the_direct_memory_route_and_dictionary_messages` | `tests/memory/test_dream.py::test_dream_processes_claimed_summaries_through_restricted_memory_route` |
| `tests/memory/test_memory_task.py::test_memory_task_direct_router_receives_tool_results_as_follow_up_dictionaries` | `tests/memory/test_dream.py::test_dream_edit_refreshes_the_manager_snapshot_after_a_successful_edit` |
| `tests/memory/test_memory_task.py::test_manual_memory_task_does_not_borrow_a_foreground_session_log` | `tests/memory/test_dream.py::test_dream_processes_claimed_summaries_through_restricted_memory_route` |
| `tests/memory/test_memory_task.py::test_memory_task_without_an_edit_advances_the_summary_cursor` | `tests/memory/test_dream.py::test_dream_processes_claimed_summaries_through_restricted_memory_route` |
| `tests/memory/test_memory_task.py::test_memory_task_advances_the_summary_cursor_before_model_work` | `tests/memory/test_dream.py::test_dream_model_failure_keeps_the_accepted_cursor` |
| `tests/memory/test_memory_task.py::test_memory_task_preadvances_summary_cursor_before_exact_edit` | `tests/memory/test_dream.py::test_dream_edit_refreshes_the_manager_snapshot_after_a_successful_edit` |
| `tests/memory/test_memory_task.py::test_memory_task_catalog_denies_every_non_long_term_memory_path` | `tests/memory/test_dream.py::test_dream_tool_failure_keeps_the_accepted_cursor_without_retry` |
| `tests/memory/test_memory_task.py::test_required_memory_edit_failure_keeps_the_advanced_summary_cursor` | `tests/memory/test_dream.py::test_dream_edit_failure_keeps_the_accepted_cursor` |
| `tests/memory/test_memory_task.py::test_unexpected_memory_tool_failure_is_logged_once_at_the_task_boundary` | `tests/memory/test_dream.py::test_dream_logs_an_unexpected_tool_failure_once_at_its_boundary` |
| `tests/memory/test_memory_task.py::test_conversation_summary_read_failure_is_logged_only_at_memory_task_boundary` | `tests/memory/test_dream.py::test_dream_logs_a_corrupt_summary_failure_once_without_leaking_content` |
| `tests/memory/test_memory_task.py::test_restricted_memory_catalog_never_reads_through_an_external_hard_link` | `tests/memory/test_dream.py::test_dream_never_reads_through_an_external_long_term_memory_hard_link` |
| `tests/memory/test_memory_task.py::test_overlapping_manual_memory_task_is_rejected_without_a_second_model_call` | `tests/memory/test_dream.py::test_dream_concurrent_runs_claim_once_and_do_not_reenter` |
| `tests/memory/test_memory_task.py::test_overlapping_manual_memory_task_ignores_a_corrupt_cursor` | `tests/memory/test_dream.py::test_dream_concurrent_runs_claim_once_and_do_not_reenter`; the injected mid-run Cursor corruption was a duplicate proof that the rejected call performs no second claim |
| `tests/memory/test_memory_task.py::test_dream_command_returns_exact_no_pending_output_without_a_model_call` | `tests/management/test_management_commands.py::test_dream_command_projects_the_complete_final_result[no-pending]` |
| `tests/memory/test_memory_task.py::test_dream_uses_memory_route_with_static_default_fallback` | `tests/memory/test_dream.py::test_dream_uses_the_memory_route_with_static_default_fallback` |
| `tests/memory/test_memory_task.py::test_dream_command_renders_model_failure_after_advancing_cursor` | `tests/management/test_management_commands.py::test_dream_command_projects_the_complete_final_result[model-failure]`; Cursor pre-advance is independently owned by `test_dream_model_failure_keeps_the_accepted_cursor` |
| `tests/memory/test_memory_task.py::test_dream_reports_cursor_publication_failure_as_unprocessed` | `tests/memory/test_dream.py::test_dream_cursor_publication_failure_is_unprocessed_and_logged_once`; result rendering is independently owned by `test_dream_command_projects_the_complete_final_result[cursor-publication-failure]` |
| `tests/memory/test_memory_task.py::test_dream_reports_corrupt_cursor_without_calling_the_model[not-a-cursor\n]` | `tests/memory/test_memory_manager.py::test_manager_rejects_corrupt_canonical_cursor_without_mutation[not-a-cursor\n]` |
| `tests/memory/test_memory_task.py::test_dream_reports_corrupt_cursor_without_calling_the_model[-1\n]` | `tests/memory/test_memory_manager.py::test_manager_rejects_corrupt_canonical_cursor_without_mutation[-1\n]` |
| `tests/memory/test_memory_task.py::test_dream_reports_corrupt_cursor_without_calling_the_model[1]` | `tests/memory/test_memory_manager.py::test_manager_rejects_corrupt_canonical_cursor_without_mutation[1]` |
| `tests/memory/test_memory_task.py::test_dream_reports_corrupt_cursor_without_calling_the_model[1 \n]` | `tests/memory/test_memory_manager.py::test_manager_rejects_corrupt_canonical_cursor_without_mutation[1 \n]` |
| `tests/memory/test_memory_task.py::test_dream_reports_corrupt_cursor_without_calling_the_model[1\n2\n]` | `tests/memory/test_memory_manager.py::test_manager_rejects_corrupt_canonical_cursor_without_mutation[1\n2\n]` |
| `tests/memory/test_memory_task.py::test_memory_task_rejects_an_external_hard_linked_cursor` | `tests/memory/test_memory_manager.py::test_manager_rejects_external_hard_linked_cursor` |
| `tests/memory/test_memory_manager.py::test_memory_task_facade_contains_only_reexports` | Legacy-only: the forwarding module and its identity assertion were deleted; source/import/wheel absence is owned by `tests/test_release_contract.py::test_standards_2_3_legacy_interfaces_are_absent_from_source` and `test_clean_distributions_omit_deleted_agent_module_and_import_cleanly` |

### Deleted REPL and headless Terminal nodes

| Deleted node | Surviving target or disposition |
| --- | --- |
| `tests/test_repl.py::test_terminal_repl_is_a_thin_compatibility_export` | Legacy-only: both forwarding modules are intentionally absent; absence is enforced by the release contracts |
| `tests/test_repl.py::test_repl_ignores_blank_and_exit_input_without_creating_inbound_messages` | Legacy-only: the plain input loop no longer exists; Textual blank and exit behavior remains covered at its product boundary |
| `tests/test_repl.py::test_repl_projects_sparse_segments_tool_arguments_and_one_terminal_marker` | `tests/terminal/test_conversation.py::test_tool_activity_renders_raw_arguments_until_terminal_marker` |
| `tests/test_repl.py::test_repl_keeps_unknown_slash_input_on_the_inbound_bus_and_dispatches_management` | `tests/terminal/test_conversation.py::test_unknown_slash_input_remains_an_ordinary_foreground_turn` |
| `tests/test_repl.py::test_repl_task_cancellation_requests_control_cancel_and_repairs_input_loop` | `tests/terminal/test_conversation.py::test_active_turn_keeps_input_editable_and_cancellable_before_a_later_turn` |
| `tests/test_repl_confirmation.py::test_repl_displays_normalized_confirmation_and_accepts_only_yes_or_no_contract` | Legacy-only: the removed line prompt accepted text; Textual owns the current button and keyboard Confirmation contract |
| `tests/test_repl_confirmation.py::test_repl_confirmation_declines_on_no_or_empty_input[no]` | Legacy-only: the removed line prompt has no product seam |
| `tests/test_repl_confirmation.py::test_repl_confirmation_declines_on_no_or_empty_input[n]` | Legacy-only: the removed line prompt has no product seam |
| `tests/test_repl_confirmation.py::test_repl_confirmation_declines_on_no_or_empty_input[]` | Legacy-only: the removed line prompt has no product seam |
| `tests/terminal/test_repl_bus.py::test_repl_preserves_tool_output_when_confirmation_is_ready_together` | `tests/terminal/test_conversation.py::test_intermediate_model_output_and_tools_share_one_activity_group` |
| `tests/terminal/test_conversation.py::test_non_tty_terminal_streams_are_rejected_before_textual_starts[stdin]` | Duplicate of the installed public entry contract `tests/test_cli.py::test_installed_myclaw_rejects_valid_configuration_without_a_tty` |
| `tests/terminal/test_conversation.py::test_non_tty_terminal_streams_are_rejected_before_textual_starts[stdout]` | Duplicate of `tests/test_cli.py::test_installed_myclaw_rejects_valid_configuration_without_a_tty` |
| `tests/terminal/test_conversation.py::test_non_tty_terminal_streams_are_rejected_before_textual_starts[stderr]` | Duplicate of `tests/test_cli.py::test_installed_myclaw_rejects_valid_configuration_without_a_tty` |
| `tests/terminal/test_conversation.py::test_non_tty_terminal_streams_are_rejected_before_textual_starts[__stdin__]` | Duplicate of `tests/test_cli.py::test_installed_myclaw_rejects_valid_configuration_without_a_tty` |
| `tests/terminal/test_conversation.py::test_non_tty_terminal_streams_are_rejected_before_textual_starts[__stdout__]` | Duplicate of `tests/test_cli.py::test_installed_myclaw_rejects_valid_configuration_without_a_tty` |
| `tests/terminal/test_conversation.py::test_non_tty_terminal_streams_are_rejected_before_textual_starts[__stderr__]` | Duplicate of `tests/test_cli.py::test_installed_myclaw_rejects_valid_configuration_without_a_tty` |

## Protected boundaries

The original Issue #201 migration did not edit its protected architecture
authorities. This convergence addendum makes only factual final-interface and
evidence updates to Runtime Contracts, Terminal design, release readiness, and
the implementation plan; it does not change ADR-0017's accepted ownership model.
