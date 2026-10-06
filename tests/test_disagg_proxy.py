"""The disaggregated proxy against scripted prefill and decode servers."""

import json

import pytest

pytest.importorskip("fastapi", reason="the serve extra is not installed")

import httpx
from fastapi.testclient import TestClient

from lean_vllm.entrypoints.disagg_proxy import build_proxy_app

PARAMS = {"do_remote_prefill": True, "remote_block_ids": [1, 2], "remote_request_id": "cmpl-p"}
SSE = 'data: {"choices": [{"text": "Hi"}]}\n\ndata: [DONE]\n\n'


class Upstreams:
    """Every server behind the proxy: records what each was sent, answers as scripted."""

    def __init__(self):
        self.sent: list[tuple[str, dict]] = []
        self.prefill_reply = httpx.Response(200, json={"choices": [{"text": "x"}], "kv_transfer_params": PARAMS})
        self.healthy = True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if request.url.path == "/health":
            return httpx.Response(200 if self.healthy else 503, json={})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": host}]})
        body = json.loads(request.content)
        self.sent.append((host, body))
        if host.startswith("prefill"):
            return self.prefill_reply
        if body.get("stream"):
            # A stream, as off the network; a response built from text= counts as read already.
            return httpx.Response(
                200, stream=httpx.ByteStream(SSE.encode()), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json={"choices": [{"text": "decoded"}], "kv_transfer_params": None})


@pytest.fixture
def upstreams():
    return Upstreams()


@pytest.fixture
def proxy(upstreams):
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstreams))
    app = build_proxy_app(["http://prefill-a", "http://prefill-b/"], ["http://decode-a"], client=client)
    with TestClient(app) as test_client:
        yield test_client


def complete(proxy, **overrides):
    body = {"model": "m", "prompt": "hi", "max_tokens": 16} | overrides
    return proxy.post("/v1/completions", json=body)


def test_the_prefill_runs_one_token_and_hands_off(proxy, upstreams):
    complete(proxy, stream=True, stream_options={"include_usage": True})
    host, body = upstreams.sent[0]
    assert host == "prefill-a"
    assert body == {
        "model": "m",
        "prompt": "hi",
        "max_tokens": 1,
        "stream": False,
        "kv_transfer_params": {"do_remote_decode": True},
    }


def test_the_decode_gets_the_request_whole_and_the_params(proxy, upstreams):
    complete(proxy, temperature=0)
    host, body = upstreams.sent[1]
    assert host == "decode-a"
    assert body == {"model": "m", "prompt": "hi", "max_tokens": 16, "temperature": 0, "kv_transfer_params": PARAMS}


def test_the_decode_s_reply_is_the_client_s(proxy):
    response = complete(proxy)
    assert response.status_code == 200 and response.json()["choices"][0]["text"] == "decoded"


def test_a_stream_is_relayed(proxy):
    response = complete(proxy, stream=True)
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == SSE


def test_chat_takes_the_same_path(proxy, upstreams):
    proxy.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert [host for host, _ in upstreams.sent] == ["prefill-a", "decode-a"]
    assert upstreams.sent[1][1]["kv_transfer_params"] == PARAMS


def test_prefill_servers_take_turns(proxy, upstreams):
    complete(proxy)
    complete(proxy)
    complete(proxy)
    assert [host for host, _ in upstreams.sent[::2]] == ["prefill-a", "prefill-b", "prefill-a"]


def test_a_prefill_error_is_the_answer(proxy, upstreams):
    upstreams.prefill_reply = httpx.Response(400, json={"error": {"message": "bad", "type": "invalid_request_error"}})
    response = complete(proxy)
    assert response.status_code == 400 and response.json()["error"]["message"] == "bad"
    assert len(upstreams.sent) == 1


def test_no_params_back_means_the_decode_prefills_itself(proxy, upstreams):
    """A prefill that stopped before its token hands nothing over."""
    upstreams.prefill_reply = httpx.Response(200, json={"choices": [{"text": ""}], "kv_transfer_params": None})
    complete(proxy)
    assert "kv_transfer_params" not in upstreams.sent[1][1]


def test_models_come_from_a_decode_server(proxy):
    assert proxy.get("/v1/models").json() == {"data": [{"id": "decode-a"}]}


def test_health_needs_every_server(proxy, upstreams):
    assert proxy.get("/health").status_code == 200
    upstreams.healthy = False
    assert proxy.get("/health").status_code == 503
