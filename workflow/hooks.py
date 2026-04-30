from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.chat import create_chat_adapter as _default_create_chat_adapter

from .editing import open_plan_in_editor as _default_open_plan_in_editor


def _default_confirm(prompt: str, *, default: bool = True) -> bool:
    from rich.prompt import Confirm

    return Confirm.ask(prompt, default=default)


@dataclass(frozen=True, slots=True)
class WorkflowHooks:
    create_chat_adapter: Callable[..., Any] = _default_create_chat_adapter
    open_plan_in_editor: Callable[[Path], Any] = _default_open_plan_in_editor
    confirm_apply_changes: Callable[..., bool] = _default_confirm
