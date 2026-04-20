from agent.context import build_context_async
from agent.events import (
    AgentEvent,
    ContextSemanticMatchItem,
    ContextSemanticMatchPayload,
    ContextSkippedPayload,
    ContextSummaryPayload,
    ContextWarningPayload,
    EventBus,
    StateChangedPayload,
)

from ._state import WorkflowRuntime, WorkflowState


async def run_build_context_stage(
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    event_bus.publish(AgentEvent(
        name="state.changed",
        state=WorkflowState.BUILDING_CONTEXT.value,
        message="Building context.",
        payload=StateChangedPayload(),
    ))
    with runtime.console.status("[bold green]Analyzing codebase..."):
        runtime.context_result = await build_context_async(
            runtime.target_dir,
            runtime.prompt,
            runtime.config,
            event_bus=event_bus,
        )
    runtime.working_context = runtime.context_result.context

    for warning in runtime.context_result.token_warnings:
        event_bus.publish(AgentEvent(
            name="context.warning",
            level="warning",
            state=WorkflowState.BUILDING_CONTEXT.value,
            message=warning,
            payload=ContextWarningPayload(warning=warning),
        ))

    if runtime.context_result.skipped_paths:
        event_bus.publish(AgentEvent(
            name="context.skipped",
            level="warning",
            state=WorkflowState.BUILDING_CONTEXT.value,
            payload=ContextSkippedPayload(
                skipped_count=len(runtime.context_result.skipped_paths),
                paths=tuple(runtime.context_result.skipped_paths),
            ),
        ))

    context_window = (
        runtime.context_result.budget_breakdown.context_window_tokens
        if runtime.context_result.budget_breakdown is not None
        else runtime.config.chat_context_window
    )
    if context_window is None:
        raise ValueError("chat_context_window must be initialized.")

    event_bus.publish(AgentEvent(
        name="context.summary",
        state=WorkflowState.BUILDING_CONTEXT.value,
        payload=ContextSummaryPayload(
            file_count=len(runtime.context_result.entries),
            used_tokens=runtime.context_result.used_tokens,
            token_budget=runtime.context_result.token_budget,
            context_window_tokens=context_window,
            response_reserve_tokens=(
                runtime.context_result.budget_breakdown.response_reserve_tokens
                if runtime.context_result.budget_breakdown is not None
                else runtime.config.max_response_tokens
            ),
            scaffold_tokens=(
                runtime.context_result.budget_breakdown.scaffold_tokens
                if runtime.context_result.budget_breakdown is not None
                else 0
            ),
            safety_margin_tokens=(
                runtime.context_result.budget_breakdown.safety_margin_tokens
                if runtime.context_result.budget_breakdown is not None
                else 0
            ),
        ),
    ))

    if runtime.context_result.semantic_matches:
        event_bus.publish(AgentEvent(
            name="context.semantic_matches",
            level="debug",
            state=WorkflowState.BUILDING_CONTEXT.value,
            message="Semantic matches selected for context.",
            payload=ContextSemanticMatchPayload(
                prompt=runtime.prompt,
                selected_count=len(runtime.context_result.semantic_matches),
                matches=tuple(
                    ContextSemanticMatchItem(
                        path=match.path,
                        semantic_score=match.semantic_score,
                        lexical_score=match.lexical_score,
                        blended_score=match.blended_score,
                    )
                    for match in runtime.context_result.semantic_matches
                ),
            ),
        ))
    return WorkflowState.PLANNING
