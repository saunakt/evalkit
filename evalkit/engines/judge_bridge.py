"""Judge pass-through for third-party metric libraries (Plan.md §12.3).

DeepEval and Ragas otherwise fall back to their own default (possibly paid) model. These
helpers expose evalkit's `judge` role as a plain `complete(prompt, schema)` function so every
library call goes through evalkit's provider, cost ledger, cache and cost cap.
"""

from __future__ import annotations

import os
from typing import Any

from evalkit.judges.runner import JudgeParseError
from evalkit.llm.client import RoleClient
from evalkit.util import extract_json

TELEMETRY_OFF = {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    "RAGAS_DO_NOT_TRACK": "true",
    "DEEPEVAL_UPDATE_WARNING_OPT_IN": "NO",
}


def disable_telemetry() -> None:
    """Library telemetry would make network calls outside the judge endpoint."""
    for key, val in TELEMETRY_OFF.items():
        os.environ[key] = val


class JudgeBridge:
    """`generate(prompt, schema)` over the judge RoleClient (schema = a pydantic model)."""

    def __init__(self, client: RoleClient, engine: str, case_id: str | None = None) -> None:
        self.client = client
        self.engine = engine
        self.case_id = case_id
        self.calls = 0

    def generate(self, prompt: str, schema: Any = None) -> Any:
        self.calls += 1
        system = "You are an evaluation model. Follow the instructions exactly."
        if schema is not None and hasattr(schema, "model_json_schema"):
            system += (
                " Respond with one JSON object matching this JSON schema: "
                f"{schema.model_json_schema()}"
            )
        resp = self.client.complete(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            response_format="json" if schema is not None else "text",
            meta={"engine": self.engine, "case_id": self.case_id, "judge_id": self.engine},
        )
        if schema is None:
            return resp.text
        data = extract_json(resp.text)
        if data is None:
            raise JudgeParseError("judge_parse_error")
        return schema.model_validate(data)
