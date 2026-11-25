"""Simulated user: an LLM plays a persona with a hidden goal (Plan.md §7.3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from evalkit.schema import Observation

DONE = "<DONE>"

SIM_SYSTEM = """You are role-playing a customer talking to a support assistant.
Persona: {persona}
Your hidden goal (never state it verbatim): {goal}
Reply with the customer's next message only. When the goal is met, or clearly
impossible, reply with exactly {done}."""


@dataclass
class SimulationResult:
    messages: list[dict[str, str]] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    user_turns: int = 0
    done: bool = False


def _flip(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """The simulator sees the assistant as 'user' and itself as 'assistant'."""
    swap = {"user": "assistant", "assistant": "user"}
    out = [{"role": swap.get(m["role"], m["role"]), "content": m["content"]} for m in messages]
    return out or [{"role": "user", "content": "(The conversation starts. Say hello.)"}]


def simulate(
    *,
    persona: str,
    hidden_goal: str,
    max_turns: int,
    llm: Any,
    call_target: Callable[[list[dict[str, str]]], Observation],
    first_message: str | None = None,
    case_id: str | None = None,
) -> SimulationResult:
    """Alternate simulator and target turns until `<DONE>` or `max_turns` user turns."""
    res = SimulationResult()
    system = SIM_SYSTEM.format(persona=persona, goal=hidden_goal, done=DONE)
    for turn in range(max_turns):
        if turn == 0 and first_message:
            user_msg = first_message
        else:
            resp = llm.complete(
                system=system,
                messages=_flip(res.messages),
                meta={"case_id": case_id, "turn": turn},
            )
            user_msg = str(resp.text).strip()
        if DONE in user_msg:
            res.done = True
            break
        res.messages.append({"role": "user", "content": user_msg})
        res.user_turns += 1
        obs = call_target(list(res.messages))
        res.observations.append(obs)
        res.messages.append({"role": "assistant", "content": obs.text})
    return res
