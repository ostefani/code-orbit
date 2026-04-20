from rich.prompt import Confirm

from agent.events import AgentEvent, EventBus, StateChangedPayload
from agent.llm import format_plan_roadmap
from agent.patcher import apply_changes, git_commit, preview_changes

from ._state import WorkflowRuntime, WorkflowState, reset_execution_state


def run_review_diff_stage(
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    assert runtime.approved_plan is not None

    event_bus.publish(AgentEvent(
        name="state.changed",
        state=WorkflowState.REVIEWING_DIFF.value,
        message="Previewing proposed changes.",
        payload=StateChangedPayload(),
    ))
    preview_changes(runtime.target_dir, runtime.all_changes, event_bus=event_bus)

    if runtime.config.interactive:
        event_bus.publish(AgentEvent(
            name="state.changed",
            state="waiting_for_user",
            message="Waiting for user confirmation.",
            payload=StateChangedPayload(),
        ))
        runtime.console.print()
        if not Confirm.ask(
            "[bold yellow]Apply these changes?[/bold yellow]",
            console=runtime.console,
        ):
            reset_execution_state(runtime)
            return WorkflowState.EDITING_PLAN

    return WorkflowState.APPLYING


def run_applying_stage(
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    event_bus.publish(AgentEvent(
        name="state.changed",
        state=WorkflowState.APPLYING.value,
        message="Applying changes.",
        payload=StateChangedPayload(),
    ))
    runtime.console.print("\n[bold green]📝 Applying changes...[/bold green]")
    affected = apply_changes(runtime.target_dir, runtime.all_changes, event_bus=event_bus)
    runtime.final_summary = runtime.final_summary or runtime.approved_plan.summary
    runtime.affected_files = affected
    return WorkflowState.COMMITTING if runtime.config.auto_commit and affected else WorkflowState.COMPLETED


def run_committing_stage(
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    affected = runtime.affected_files
    if affected:
        event_bus.publish(AgentEvent(
            name="state.changed",
            state=WorkflowState.COMMITTING.value,
            message="Creating git commit.",
            payload=StateChangedPayload(),
        ))
        git_commit(
            runtime.target_dir,
            f"{runtime.final_summary}\n\n{format_plan_roadmap(runtime.approved_plan)}",
            affected,
            event_bus=event_bus,
        )
    return WorkflowState.COMPLETED
