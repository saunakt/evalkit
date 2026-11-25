from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evalkit.config import EvalkitConfig, Price, ProviderConfig, RoleModel
from evalkit.env.cache import CacheMiss, ResponseCache
from evalkit.llm.anthropic_client import AnthropicClient
from evalkit.llm.client import (
    CostCapReached,
    CostLedger,
    LLMResponse,
    RetryableError,
    RoleClient,
    with_retries,
)
from evalkit.llm.fake import FakeLLM
from evalkit.llm.openai_compatible import OpenAICompatibleClient, parse_sse_line
from evalkit.llm.roles import LLMRuntime, client_for, is_free, set_runtime
from tests.conftest import FakeServer, Reply, chat_reply

MSG = [{"role": "user", "content": "hello"}]


def oa(server: FakeServer, **kw: Any) -> OpenAICompatibleClient:
    cfg = ProviderConfig(type="openai_compatible", base_url=server.url, retry_base_s=0.01, **kw)
    return OpenAICompatibleClient("local", cfg)


# ------------------------------------------------------------------ openai-compatible


def test_openai_normal_reply_with_usage(http_server: FakeServer) -> None:
    http_server.queue(Reply(body=chat_reply("hi", {"prompt_tokens": 7, "completion_tokens": 2})))
    resp = oa(http_server, api_key="k").complete(system="sys", messages=MSG, model="m")
    assert (resp.text, resp.tokens_in, resp.tokens_out, resp.tokens_estimated) == (
        "hi",
        7,
        2,
        False,
    )
    req = http_server.requests[0]
    assert req["path"] == "/chat/completions"
    assert req["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert req["headers"]["Authorization"] == "Bearer k"
    assert "response_format" not in req["body"]


def test_openai_json_mode_only_when_supported(http_server: FakeServer) -> None:
    http_server.queue(Reply(body=chat_reply("{}")), Reply(body=chat_reply("{}")))
    oa(http_server, supports_json=True).complete(
        system="", messages=MSG, model="m", response_format="json"
    )
    oa(http_server).complete(system="", messages=MSG, model="m", response_format="json")
    assert http_server.requests[0]["body"]["response_format"] == {"type": "json_object"}
    assert "response_format" not in http_server.requests[1]["body"]


def test_openai_missing_usage_is_estimated(http_server: FakeServer) -> None:
    http_server.queue(Reply(body=chat_reply("abcdefgh")))
    resp = oa(http_server).complete(system="", messages=MSG, model="m")
    assert resp.tokens_estimated and resp.tokens_out == 2


def test_openai_retries_429_then_succeeds(http_server: FakeServer) -> None:
    http_server.queue(Reply(status=429), Reply(status=503), Reply(body=chat_reply("ok")))
    assert oa(http_server).complete(system="", messages=MSG, model="m").text == "ok"
    assert len(http_server.requests) == 3


def test_openai_timeout_raises_after_retries(http_server: FakeServer) -> None:
    http_server.default = Reply(body=chat_reply("late"), delay_s=0.5)
    client = oa(http_server, timeout_s=0.1, max_retries=2)
    with pytest.raises(RetryableError):
        client.complete(system="", messages=MSG, model="m")


def test_openai_sse_stream(http_server: FakeServer) -> None:
    deltas = [json.dumps({"choices": [{"delta": {"content": c}}]}) for c in ("Hel", "lo")]
    http_server.queue(Reply(sse=[*deltas, "[DONE]"]))
    assert "".join(oa(http_server).stream(system="", messages=MSG, model="m")) == "Hello"
    assert http_server.requests[0]["body"]["stream"] is True


def test_parse_sse_line() -> None:
    assert parse_sse_line("event: ping") == ""
    assert parse_sse_line("data: [DONE]") is None


def test_with_retries_gives_up() -> None:
    calls: list[int] = []

    def boom() -> None:
        calls.append(1)
        raise RetryableError("x")

    with pytest.raises(RetryableError):
        with_retries(boom, max_tries=3, sleep=lambda _: None)
    assert len(calls) == 3


# ------------------------------------------------------------------ anthropic (optional)


def test_anthropic_wrapper_with_stub_sdk() -> None:
    created: dict[str, Any] = {}

    class Messages:
        def create(self, **kw: Any) -> Any:
            created.update(kw)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="PASS")],
                usage=SimpleNamespace(input_tokens=11, output_tokens=3),
                model=kw["model"],
            )

    sdk = SimpleNamespace(Anthropic=lambda **kw: SimpleNamespace(messages=Messages()))
    client = AnthropicClient("anthropic", ProviderConfig(type="anthropic", api_key="k"), sdk=sdk)
    resp = client.complete(system="s", messages=MSG, model="m", temperature=0.0)
    assert (resp.text, resp.tokens_in, resp.tokens_out) == ("PASS", 11, 3)
    assert "temperature" not in created and created["system"] == "s"


def test_anthropic_real_sdk_constructs_if_installed() -> None:
    pytest.importorskip("anthropic")
    AnthropicClient("anthropic", ProviderConfig(type="anthropic", api_key="test"))


# ------------------------------------------------------------------ fake provider


def test_fake_rules_judge_script_and_defaults() -> None:
    fake = FakeLLM(
        script={
            "rules": [{"match": "refund", "response": "30 days"}],
            "judge_script": [{"case": "c1", "judge": "j", "verdict": "FAIL"}],
        }
    )
    assert (
        fake.complete(system="", messages=[{"role": "user", "content": "refund?"}], model="f").text
        == "30 days"
    )
    assert fake.complete(system="", messages=MSG, model="f").text == "hello"
    verdict = fake.complete(
        system="", messages=MSG, model="f", meta={"role": "judge", "case_id": "c1", "judge_id": "j"}
    )
    assert json.loads(verdict.text)["verdict"] == "FAIL"
    default = fake.complete(
        system="", messages=MSG, model="f", meta={"role": "judge", "style": "rubric"}
    )
    assert json.loads(default.text)["score"] == 5
    sim = fake.complete(system="", messages=MSG, model="f", meta={"role": "simulator"})
    assert sim.text == "<DONE>"


# ------------------------------------------------------------------ ledger / role client


def _role_client(
    free: bool, price: Price | None, cap: float | None = None, cache: ResponseCache | None = None
) -> RoleClient:
    return RoleClient(
        role="judge",
        provider_name="p",
        client=FakeLLM(),
        model="m",
        free=free,
        price=price,
        ledger=CostLedger(),
        cache=cache,
        max_cost_usd=cap,
    )


def test_ledger_paid_vs_free_and_pricing() -> None:
    paid = _role_client(False, Price(input_per_mtok=1_000_000, output_per_mtok=0))
    resp = paid.complete(system="", messages=[{"role": "user", "content": "abcd"}])
    assert resp.cost_usd == pytest.approx(resp.tokens_in)
    s = paid.ledger.summary()["judge"]
    assert s["paid_calls"] == 1 and s["free_calls"] == 0
    free = _role_client(True, None)
    assert free.complete(system="", messages=MSG).cost_usd == 0.0
    assert free.ledger.summary()["judge"]["free_calls"] == 1


def test_missing_price_records_null_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    rc = _role_client(False, None)
    with caplog.at_level(logging.WARNING, logger="evalkit.llm"):
        rc.complete(system="", messages=MSG)
        rc.complete(system="", messages=[{"role": "user", "content": "x"}])
    assert rc.ledger.summary()["judge"]["unpriced_paid_calls"] == 2
    assert sum("no price" in r.message for r in caplog.records) == 1


def test_cost_cap_blocks_paid_calls() -> None:
    rc = _role_client(False, Price(input_per_mtok=1e6, output_per_mtok=1e6), cap=0.5)
    rc.complete(system="", messages=MSG)
    with pytest.raises(CostCapReached):
        rc.complete(system="", messages=[{"role": "user", "content": "again"}])


# ------------------------------------------------------------------ cache


def test_cache_modes(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    rw = ResponseCache(path, "read_write")
    key = rw.key("p", "m", "s", MSG, 0.0, 10)
    assert rw.get(key) is None
    rw.put(key, LLMResponse(text="cached").model_dump())
    assert rw.get(key)["text"] == "cached"  # type: ignore[index]
    rw.close()
    off = ResponseCache(path, "off")
    assert off.get(key) is None
    replay = ResponseCache(path, "replay_only")
    assert replay.get(key) is not None
    with pytest.raises(CacheMiss):
        replay.get("missing")


def test_role_client_uses_cache(tmp_path: Path) -> None:
    rc = _role_client(True, None, cache=ResponseCache(tmp_path / "c.sqlite"))
    rc.complete(system="", messages=MSG)
    second = rc.complete(system="", messages=MSG)
    assert second.cached and rc.ledger.summary()["judge"]["cached_calls"] == 1


# ------------------------------------------------------------------ roles


def test_is_free_rules() -> None:
    assert is_free(ProviderConfig(type="openai_compatible", base_url="http://localhost:11434/v1"))
    assert not is_free(ProviderConfig(type="openai_compatible", base_url="https://x.ai/v1"))
    assert is_free(ProviderConfig(type="fake"))
    assert not is_free(ProviderConfig(type="fake", free=False))


def test_client_for_resolves_role(tmp_path: Path) -> None:
    cfg = EvalkitConfig(
        providers={"fake": ProviderConfig(type="fake")},
        models={"judge": RoleModel(provider="fake", model="fj"), "sut": None},
    )
    rt = LLMRuntime(cfg, tmp_path)
    set_runtime(rt)
    try:
        client, model = client_for("judge")
        assert model == "fj" and client.free
        client.complete(system="", messages=MSG)
        assert rt.ledger.summary()["judge"]["calls"] == 1
    finally:
        set_runtime(None)
