"""multimodal — voice, images and documents (ch. 29.1–29.3)."""

from __future__ import annotations

from evalkit.registry import suite
from evalkit.schema import Case, Observation
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext


@suite("multimodal")
class MultimodalSuite(BaseSuite):
    """Voice, images and documents (ch. 29.1–29.3).

    Media paths are relative to datasets/; targets must set `supports_media: true` to receive
    them (as `ctx.params.media_path`). Otherwise every metric of the case is skipped cleanly.
    """

    name = "multimodal"
    default_metrics = (
        "wer",
        "entity_accuracy",
        "voice_latency",
        "field_accuracy",
        "refuses_unreadable",
        "visual_hallucination",
    )

    def skip_reason(self, case: Case, sctx: SuiteContext) -> str | None:
        if case.media and not getattr(sctx.target, "supports_media", False):
            return "target cannot process media (supports_media: false)"
        return None

    def execute(self, case: Case, repeat: int, sctx: SuiteContext) -> Observation:
        if case.media:
            path = sctx.project.path("datasets") / case.media
            return sctx.call(case, params={"media_path": str(path)})
        return super().execute(case, repeat, sctx)
