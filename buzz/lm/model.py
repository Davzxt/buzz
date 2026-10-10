"""Buzz-LM: Transformer decoder próprio (RoPE + RMSNorm + SwiGLU + embeddings amarrados + KV-cache).
~20M parâmetros na config padrão; pensado para treinar do zero e inferir na CPU de uma VPS pequena."""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    vocab: int = 16000
    dim: int = 384
    layers: int = 8
    heads: int = 6
    ctx: int = 512
    hidden: int = 1024


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.h, self.hd = c.heads, c.dim // c.heads
        self.n1, self.n2 = RMSNorm(c.dim), RMSNorm(c.dim)
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.o = nn.Linear(c.dim, c.dim, bias=False)
        self.gate = nn.Linear(c.dim, c.hidden, bias=False)
        self.up = nn.Linear(c.dim, c.hidden, bias=False)
        self.down = nn.Linear(c.hidden, c.dim, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        if cache is not None:
            if cache:
                k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
            cache[:] = [k, v]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class GPT(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.dim)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = RMSNorm(c.dim)
        hd = c.dim // c.heads
        f = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.outer(torch.arange(c.ctx).float(), f)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        self.apply(self._init)
        for b in self.blocks:  # init residual reduzido
            nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * c.layers))
            nn.init.normal_(b.down.weight, std=0.02 / math.sqrt(2 * c.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None, pos=0):
        T = idx.shape[1]
        x = self.emb(idx)
        cos, sin = self.cos[pos:pos + T], self.sin[pos:pos + T]
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, None if caches is None else caches[i])
        logits = F.linear(self.norm(x), self.emb.weight)  # pesos amarrados
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=128, temperature=0.8, top_k=40, stop_id=None, rep_penalty=1.1):
        caches = [[] for _ in self.blocks]
        logits, _ = self(idx[:, -self.c.ctx:], caches=caches)
        out, pos = [], idx.shape[1]
        seen = set(idx[0].tolist())
        for _ in range(max_new):
            if pos >= self.c.ctx:
                break
            lg = logits[0, -1].float()
            for t in seen:
                lg[t] = lg[t] / rep_penalty if lg[t] > 0 else lg[t] * rep_penalty
            if temperature <= 0:
                nxt = int(lg.argmax())
            else:
                lg = lg / temperature
                if top_k:
                    lg[lg < torch.topk(lg, top_k).values[-1]] = -float("inf")
                nxt = int(torch.multinomial(F.softmax(lg, -1), 1))
            if nxt == stop_id:
                break
            out.append(nxt)
            seen.add(nxt)
            logits, _ = self(torch.tensor([[nxt]], device=idx.device), caches=caches, pos=pos)
            pos += 1
        return out


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = GPT(Cfg(**ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


def cfg_dict(c):
    return asdict(c)
