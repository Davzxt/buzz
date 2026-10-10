"""Roteador adaptativo: decide o caminho mais barato que atende a solicitação."""
import os

from .decision import Decision
from .memory import Memory, is_explicit_remember, strip_remember
from .tools import calc

SYSTEM = ("Você é o Buzz, um assistente em português do Brasil: direto, simpático e conciso. "
          "Se não souber ou não tiver certeza, diga isso claramente; nunca invente fatos, números ou fontes.")
CONF_MIN = float(os.environ.get("BUZZ_CONF_MIN", "0.55"))
HIST_CHARS = int(os.environ.get("BUZZ_HIST_CHARS", "3000"))


class Router:
    def __init__(self, decision: Decision, memory: Memory, llm):
        self.decision, self.memory, self.llm = decision, memory, llm

    def _llm(self, history, evidence, max_tokens, temperature, strict=False):
        sys = SYSTEM
        if evidence:
            sys += "\n\nEVIDÊNCIAS (podem estar incompletas):\n" + "\n".join(f"- [{e['source']}] {e['text']}" for e in evidence)
            sys += "\nBaseie-se nelas quando relevantes e diga se elas não bastam."
        if strict:
            sys += "\nResponda SOMENTE com base nas evidências. Se não bastarem, diga que não sabe."
        msgs, used = [], 0
        for m in reversed(history):  # mantém o fim da conversa dentro do orçamento
            used += len(m["content"])
            if used > HIST_CHARS and msgs:
                break
            msgs.append(m)
        text, usage = self.llm.chat([{"role": "system", "content": sys}] + msgs[::-1], max_tokens, temperature)
        return text, usage

    def handle(self, messages, max_tokens=256, temperature=0.3):
        history = [m for m in messages if m["role"] in ("user", "assistant")]
        if not history or history[-1]["role"] != "user":
            raise ValueError("a última mensagem deve ser do usuário")
        q = history[-1]["content"]
        label, conf = self.decision.predict(q)
        meta = {"label": label, "confidence": round(conf, 3), "llm": False, "evidence": 0}

        if label in ("chat", "factual") or conf < CONF_MIN:  # rede de segurança: conta válida => ferramenta
            r = calc(q)
            if r:
                return r, {**meta, "label": "math", "route": "tool:calc(guard)"}
        if conf >= CONF_MIN:
            if label == "math":
                r = calc(q)
                if r:
                    return r, {**meta, "route": "tool:calc"}
            elif label == "remember" and is_explicit_remember(q):
                self.memory.add(strip_remember(q), "user")
                return "Anotado! Vou lembrar disso.", {**meta, "route": "memory:write"}
            elif label == "recall":
                ev = self.memory.search(q)
                if not ev:
                    return "Não encontrei nada salvo sobre isso.", {**meta, "route": "memory:miss"}
                text, u = self._llm(history, ev, max_tokens, 0.1, strict=True)
                return text, {**meta, "route": "memory:llm", "llm": True, "evidence": len(ev), "usage": u}

        ev = self.memory.search(q) if label in ("factual", "summarize", "chat") or conf < CONF_MIN else []
        route = "llm" if conf >= CONF_MIN else "llm:low_conf"
        text, u = self._llm(history, ev, max_tokens, temperature)
        return text, {**meta, "route": route, "llm": True, "evidence": len(ev), "usage": u}
