import pytest
from fastapi.testclient import TestClient

from buzz.api import create_app
from buzz.data import splits
from buzz.decision import Decision
from buzz.memory import Memory, strip_remember
from buzz.router import Router
from buzz.tools import calc


class StubLLM:
    """Dublê SOMENTE para testes: registra se o LLM foi chamado e o que recebeu."""
    def __init__(self):
        self.calls = []

    def chat(self, messages, max_tokens=256, temperature=0.3):
        self.calls.append(messages)
        return "resposta-llm", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


@pytest.fixture(scope="module")
def decision():
    tr, va, _ = splits(200)
    return Decision().fit(tr, va)


@pytest.fixture
def parts(decision):
    llm = StubLLM()
    return Router(decision, Memory(), llm), llm


def test_calc():
    assert calc("quanto é 12 mais 30?") == "12 + 30 = 42"
    assert calc("calcule 7 * (3 + 4)").endswith("= 49")
    assert calc("quanto é 10 dividido por 4") .endswith("2,5")
    assert calc("oi tudo bem") is None
    assert calc("2 ** 99999") is None and calc("1/0") is None


def test_memory_provenance_and_fts():
    m = Memory()
    m.add("meu gato se chama Tom", "user")
    m.add("o gato é um felino", "generated")
    r = m.search("qual o nome do gato?")
    assert [x["source"] for x in r] == ["user"]  # 'generated' nunca vira evidência
    assert strip_remember("lembre que meu gato se chama Tom") == "meu gato se chama Tom"


def test_math_skips_llm(parts):
    r, llm = parts
    text, meta = r.handle([{"role": "user", "content": "quanto é 15 vezes 4?"}])
    assert "60" in text and meta["route"] == "tool:calc" and not llm.calls


def test_remember_then_recall(parts):
    r, llm = parts
    _, m1 = r.handle([{"role": "user", "content": "lembre que meu gato se chama Tom"}])
    assert m1["route"] == "memory:write" and not llm.calls
    _, m2 = r.handle([{"role": "user", "content": "qual é o nome do meu gato?"}])
    assert m2["route"] == "memory:llm" and "Tom" in llm.calls[-1][0]["content"]
    text, m3 = r.handle([{"role": "user", "content": "o que eu te disse sobre minha alergia?"}])
    assert m3["route"] == "memory:miss" and len(llm.calls) == 1


def test_no_memory_write_without_explicit_command_and_no_irrelevant_evidence(parts):
    r, llm = parts
    prob = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins "
            "for her friends every day with four. How much does she make selling the rest at $2 each?")
    r.handle([{"role": "user", "content": prob}])
    assert not r.memory.search("eggs")  # nada foi gravado
    r.memory.add("meu gato se chama Tom", "user")
    r.handle([{"role": "user", "content": "Explain how a bakery sells muffins to friends every day at breakfast time"}])
    assert "EVIDÊNCIAS" not in llm.calls[-1][0]["content"]


def test_decision_generalizes_to_unseen_templates(decision):
    _, _, te = splits(200)
    acc = sum(decision.predict(x["text"])[0] == x["label"] for x in te) / len(te)
    assert acc > 0.8, acc


def test_api_auth_ratelimit_and_openai_shape(parts, monkeypatch):
    monkeypatch.setenv("BUZZ_API_KEY", "k")
    monkeypatch.setenv("BUZZ_RATE_PER_MIN", "3")
    c = TestClient(create_app(parts[0]))
    body = {"messages": [{"role": "user", "content": "quanto é 2 + 2?"}]}
    assert c.post("/v1/chat/completions", json=body).status_code == 401
    h = {"Authorization": "Bearer k"}
    r = c.post("/v1/chat/completions", json=body, headers=h)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "2 + 2 = 4"
    assert c.post("/v1/chat/completions", json={"messages": [{"role": "assistant", "content": "x"}]}, headers=h).status_code == 400
    assert c.post("/v1/chat/completions", json=body, headers=h).status_code in (200, 429)
    assert c.post("/v1/chat/completions", json=body, headers=h).status_code == 429
