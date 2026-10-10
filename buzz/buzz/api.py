"""API HTTP compatível com /v1/chat/completions (sem depender de nenhum serviço externo)."""
import collections
import hmac
import logging
import os
import threading
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .decision import Decision
from .llm import LLMUnavailable, make_llm
from .memory import Memory
from .router import Router

log = logging.getLogger("buzz")
logging.basicConfig(level=os.environ.get("BUZZ_LOG", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


class Msg(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class ChatReq(BaseModel):
    model: str = "buzz"
    messages: list[Msg] = Field(min_length=1, max_length=30)
    max_tokens: int = Field(256, ge=1)
    temperature: float = Field(0.3, ge=0, le=1.5)


class MemReq(BaseModel):
    text: str = Field(max_length=2000)
    source: str = "retrieved"


def create_app(router: Router | None = None) -> FastAPI:
    key = os.environ.get("BUZZ_API_KEY", "")
    if not key and os.environ.get("BUZZ_ALLOW_NOAUTH") != "1":
        raise RuntimeError("defina BUZZ_API_KEY (ou BUZZ_ALLOW_NOAUTH=1 apenas em desenvolvimento)")
    max_tok = int(os.environ.get("BUZZ_MAX_TOKENS", "512"))
    rate = int(os.environ.get("BUZZ_RATE_PER_MIN", "20"))
    if router is None:
        router = Router(Decision.load(os.environ.get("BUZZ_DECISION_PATH", "models/decision.joblib")),
                        Memory(os.environ.get("BUZZ_DB", "buzz.db")), make_llm())
    hits, hlock = collections.defaultdict(collections.deque), threading.Lock()
    app = FastAPI(title="Buzz", docs_url=None, redoc_url=None)

    def guard(request: Request, authorization: str = Header(default="")):
        if key and not hmac.compare_digest(authorization, f"Bearer {key}"):
            raise HTTPException(401, "não autorizado")
        ip, now = request.client.host if request.client else "?", time.time()
        with hlock:
            q = hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= rate:
                raise HTTPException(429, "limite de requisições excedido")
            q.append(now)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions", dependencies=[Depends(guard)])
    def chat(req: ChatReq):
        t0 = time.time()
        try:
            text, meta = router.handle([m.model_dump() for m in req.messages], min(req.max_tokens, max_tok), req.temperature)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except LLMUnavailable as e:
            raise HTTPException(503, str(e))
        ms = round((time.time() - t0) * 1000)
        log.info("route=%s label=%s conf=%s llm=%s ms=%d tokens=%s", meta["route"], meta["label"], meta["confidence"],
                 meta["llm"], ms, meta.get("usage", {}).get("completion_tokens", 0))  # nunca loga conteúdo
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
                "model": "buzz", "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": text}}],
                "usage": meta.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "buzz": {**{k: v for k, v in meta.items() if k != "usage"}, "latency_ms": ms}}

    @app.post("/v1/memory", dependencies=[Depends(guard)])
    def mem_add(r: MemReq):
        if r.source not in ("user", "retrieved"):
            raise HTTPException(400, "source deve ser user ou retrieved")
        return {"id": router.memory.add(r.text, r.source)}

    @app.get("/v1/memory/search", dependencies=[Depends(guard)])
    def mem_search(q: str, k: int = 3):
        return {"results": router.memory.search(q, min(k, 10))}

    return app
