"""Núcleo linguístico: GGUF quantizado via llama.cpp (CPU). Carregamento preguiçoso, 1 geração por vez."""
import os
import threading


class LLMUnavailable(RuntimeError):
    pass


class OwnLM:
    """Backend do modelo PRÓPRIO (lm/): PyTorch CPU, KV-cache, formato <usuario> ... <buzz>."""

    def __init__(self, run=None):
        self.path = run or os.environ.get("BUZZ_LM_RUN", "runs/buzz-chat")
        self.n_threads = int(os.environ.get("BUZZ_THREADS", "1"))
        self.queue_timeout = float(os.environ.get("BUZZ_QUEUE_TIMEOUT", "30"))
        self._m, self._lock, self._load = None, threading.Lock(), threading.Lock()

    def _model(self):
        with self._load:
            if self._m is None:
                if not os.path.exists(f"{self.path}/best.pt"):
                    raise LLMUnavailable(f"checkpoint não encontrado em {self.path}")
                import torch
                from tokenizers import Tokenizer

                from lm.model import load
                torch.set_num_threads(self.n_threads)
                self._m, self._tok = load(f"{self.path}/best.pt"), Tokenizer.from_file(f"{self.path}/tokenizer.json")
            return self._m, self._tok

    def chat(self, messages, max_tokens=256, temperature=0.3):
        import torch
        if not self._lock.acquire(timeout=self.queue_timeout):
            raise LLMUnavailable("fila cheia, tente novamente")
        try:
            m, tok = self._model()
            sys = " ".join(x["content"] for x in messages if x["role"] == "system")
            user = [x for x in messages if x["role"] != "system"][-1]["content"]
            prompt = (f"{sys}\n\n" if "EVIDÊNCIAS" in sys else "") + user
            ids = [tok.token_to_id("<usuario>")] + tok.encode(prompt).ids + [tok.token_to_id("<buzz>")]
            ids = ids[-(m.c.ctx - max_tokens):] if len(ids) > m.c.ctx - max_tokens else ids
            out = m.generate(torch.tensor([ids]), max_tokens, max(temperature, 0.0), stop_id=tok.token_to_id("<eos>"))
        finally:
            self._lock.release()
        return tok.decode(out).strip(), {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}


def make_llm():
    return OwnLM() if os.environ.get("BUZZ_BACKEND", "own") == "own" else LLM()


class LLM:
    def __init__(self, path=None, n_ctx=None, n_threads=None):
        self.path = path or os.environ.get("BUZZ_MODEL_PATH", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        self.n_ctx = n_ctx or int(os.environ.get("BUZZ_N_CTX", "2048"))
        self.n_threads = n_threads or int(os.environ.get("BUZZ_THREADS", "1"))
        self.queue_timeout = float(os.environ.get("BUZZ_QUEUE_TIMEOUT", "30"))
        self._m, self._load_lock, self._gen_lock = None, threading.Lock(), threading.Lock()

    def _model(self):
        with self._load_lock:
            if self._m is None:
                if not os.path.exists(self.path):
                    raise LLMUnavailable(f"modelo não encontrado: {self.path} (rode scripts/fetch_model.sh)")
                from llama_cpp import Llama  # import tardio: API sobe sem o pacote se só usar o caminho rápido
                self._m = Llama(model_path=self.path, n_ctx=self.n_ctx, n_threads=self.n_threads,
                                use_mmap=True, verbose=False)
            return self._m

    def chat(self, messages, max_tokens=256, temperature=0.3):
        if not self._gen_lock.acquire(timeout=self.queue_timeout):
            raise LLMUnavailable("fila cheia, tente novamente")
        try:
            out = self._model().create_chat_completion(messages=messages, max_tokens=max_tokens,
                                                       temperature=temperature, top_p=0.9, repeat_penalty=1.1)
        finally:
            self._gen_lock.release()
        return out["choices"][0]["message"]["content"].strip(), out.get("usage", {})
