"""skills — agent skills: triggering, execution steps, output quality, with/without (ch. 23)."""

from __future__ import annotations

from evalkit.registry import suite
from evalkit.suites.base import BaseSuite


@suite("skills")
class SkillsSuite(BaseSuite):
    """Agent skills: triggering, execution steps, output quality, with/without (ch. 23).

    Targets report `skill_loaded` events (or `Skill` tool calls); `ctx.params.skill_enabled`
    toggles the skill for the with/without comparison.
    """

    name = "skills"
    default_metrics = (
        "trigger_rate",
        "false_trigger_rate",
        "execution_steps",
        "output_quality",
        "with_without_delta",
        "skill_conflict",
        "third_party_scan",
    )
