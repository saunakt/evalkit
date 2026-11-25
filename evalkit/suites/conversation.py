"""conversation — multi-turn conversations: scripted turns and simulated users (ch. 16)."""

from __future__ import annotations

from evalkit.registry import suite
from evalkit.suites.base import BaseSuite


@suite("conversation")
class ConversationSuite(BaseSuite):
    """Multi-turn conversations: scripted turns and simulated users (ch. 16).

    Scripted cases (`turns`) send the full history every turn; simulated cases
    (`persona` + `hidden_goal`) are driven by the simulator role (BaseSuite.execute).
    """

    name = "conversation"
    default_metrics = (
        "knowledge_retention",
        "reference_resolution",
        "consistency",
        "role_adherence",
        "conversation_completeness",
        "turn_relevancy",
        "turns_to_resolution",
        "context_overflow",
    )
