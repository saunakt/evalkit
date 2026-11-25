from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from evalkit.schema import Case
from evalkit.targets.base import TargetContext, coerce_observation, has_hook, load_target
from evalkit.targets.http_target import HttpTarget
from tests.conftest import ROOT, FakeServer, Reply


def test_python_target_calls_entry_and_hooks() -> None:
    t = load_target(
        {
            "type": "python",
            "entry": "tests.fixtures.sample_target:ask",
            "reset": "tests.fixtures.sample_target:reset",
            "state": "tests.fixtures.sample_target:state",
        },
        ROOT,
    )
    assert has_hook(t, "reset") and has_hook(t, "state") and not has_hook(t, "ingest")
    t.reset(7)
    obs = t.call(Case(id="a", input="hi"), TargetContext())
    assert obs.text == "echo: hi" and obs.tool_calls[0].name == "lookup"
    assert obs.latency_s > 0
    assert t.state() == {"seed": 7, "last": "hi"}


def test_python_target_messages_and_str_return() -> None:
    t = load_target({"type": "python", "entry": "tests.fixtures.sample_target:ask"}, ROOT)
    ctx = TargetContext(messages=[{"role": "user", "content": "first"}])
    assert t.call(Case(id="a", input="second"), ctx).text == "echo: second"
    t2 = load_target({"type": "python", "entry": "tests.fixtures.sample_target:ask_str"}, ROOT)
    assert (
        t2.call(Case(id="b", input="x"), TargetContext(now="2026-01-01")).text == "now=2026-01-01"
    )


def test_python_target_freeze_time() -> None:
    t = load_target({"type": "python", "entry": "datetime:date.today", "freeze_time": True}, ROOT)
    t.fn = lambda payload, ctx: __import__("datetime").date.today().isoformat()
    obs = t.call(Case(id="a", input="x"), TargetContext(now="2026-11-25T10:00:00+05:30"))
    assert obs.text == "2026-11-25"


def test_coerce_observation_rejects_bad_types() -> None:
    assert coerce_observation("x").text == "x"
    with pytest.raises(TypeError):
        coerce_observation(42)


def test_http_target_body_template_and_mapping(http_server: FakeServer) -> None:
    http_server.queue(
        Reply(body={"data": {"answer": "42", "trace": {"tools": [{"name": "calc"}]}}})
    )
    t = HttpTarget(
        {
            "url": http_server.url + "/chat",
            "headers": {"X-Key": "${EVALKIT_TEST_KEY}"},
            "body": '{"q": {{ input | tojson }}, "lang": {{ case.language | tojson }}}',
            "response": {"text": "data.answer", "tool_calls": "data.trace.tools"},
        },
        ROOT,
    )
    obs = t.call(Case(id="a", input="what?", language="en"), TargetContext())
    assert obs.text == "42" and obs.tool_calls[0].name == "calc"
    assert http_server.requests[0]["body"] == {"q": "what?", "lang": "en"}


def test_http_target_retries_and_errors(http_server: FakeServer) -> None:
    http_server.queue(Reply(status=429), Reply(status=500, body={"err": "x"}))
    t = HttpTarget({"url": http_server.url}, ROOT)
    obs = t.call(Case(id="a", input="x"), TargetContext())
    assert obs.error == "HTTP 500" and len(http_server.requests) == 2


def test_http_target_sse_stream(http_server: FakeServer) -> None:
    deltas = [json.dumps({"choices": [{"delta": {"content": c}}]}) for c in ("a", "b")]
    http_server.queue(Reply(sse=[*deltas, "[DONE]"]))
    t = HttpTarget({"url": http_server.url, "stream": "sse"}, ROOT)
    assert has_hook(t, "stream")
    assert list(t.stream(Case(id="a", input="x"), TargetContext())) == ["a", "b"]


def test_http_reset_and_state_urls(http_server: FakeServer) -> None:
    http_server.queue(Reply(body={}), Reply(body={"orders": 1}))
    t = HttpTarget(
        {
            "url": http_server.url,
            "reset_url": http_server.url + "/reset",
            "state_url": http_server.url + "/state",
        },
        ROOT,
    )
    t.reset(3)
    assert t.state() == {"orders": 1}
    assert http_server.requests[0]["body"] == {"seed": 3}


def test_cli_target_json_parse(tmp_path: Path) -> None:
    script = tmp_path / "agent.py"
    script.write_text(
        "import json,sys\n"
        "p=open(sys.argv[1],encoding='utf-8').read()\n"
        "print(json.dumps({'result': 'did: '+p, 'tokens_in': 5}))\n",
        encoding="utf-8",
    )
    t = load_target(
        {"type": "cli", "command": "{python} " + str(script) + " {prompt_file}", "parse": "json"},
        tmp_path,
    )
    obs = t.call(Case(id="a", input="fix bug"), TargetContext())
    assert obs.text == "did: fix bug" and obs.tokens_in == 5 and obs.error is None


def test_cli_target_nonzero_exit(tmp_path: Path) -> None:
    cmd = f'"{sys.executable}" -c "import sys; sys.exit(3)"'
    t = load_target({"type": "cli", "command": cmd}, tmp_path)
    assert "exit 3" in (t.call(Case(id="a", input="x"), TargetContext()).error or "")


def test_fake_target_responses_and_state() -> None:
    t = load_target(
        {
            "type": "fake",
            "responses": {"a": {"text": "scripted", "blocked": True}},
            "states": {"a": {"x": 1}},
        },
        ROOT,
    )
    assert t.call(Case(id="a", input="q"), TargetContext()).blocked is True
    assert t.state() == {"x": 1}
    assert t.call(Case(id="b", input="echo me"), TargetContext()).text == "echo me"
