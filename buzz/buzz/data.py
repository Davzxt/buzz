"""Gerador de dados sintéticos para o caminho de decisão rápida.

Três piscinas DISJUNTAS de templates (treino / validação-calibração / teste), para que a
avaliação meça generalização para formulações nunca vistas e não memorização.
Atenção: dados sintéticos => números otimistas. Para produção, substitua/complemente
com mensagens reais anotadas (data/real.jsonl, mesmo formato: {"text","label"}).
"""
import itertools
import json
import random
from pathlib import Path

LABELS = ["chat", "math", "remember", "recall", "summarize", "factual"]

NUM = ["2", "7", "12", "15", "36", "48", "100", "250", "3,5", "81", "9", "64", "1000", "0,5"]
OP = ["+", "-", "*", "/", "mais", "menos", "vezes", "dividido por"]
FACT = ["meu nome é Ana", "minha senha do wifi é azul42", "a reunião é na quinta às 15h", "meu gato se chama Tom",
        "o prazo do projeto é 12 de março", "eu moro em Curitiba", "meu time é o Flamengo", "o código da porta é 4821",
        "prefiro café sem açúcar", "minha alergia é a amendoim", "o servidor fica em São Paulo", "meu aniversário é em maio"]
TOPIC = ["a fotossíntese", "a Revolução Francesa", "o buraco negro", "a inflação", "o Python", "o protocolo HTTP",
         "a mitocôndria", "a Lua", "o Rio Amazonas", "a teoria da relatividade", "o DNA", "a Segunda Guerra Mundial"]
ASK = ["meu nome", "minha senha do wifi", "quando é a reunião", "o nome do meu gato", "o prazo do projeto",
       "onde eu moro", "qual é o meu time", "o código da porta", "como eu gosto do café", "a minha alergia"]
TEXT = ["o texto abaixo", "esse parágrafo", "o artigo que colei", "a conversa anterior", "essa notícia", "o e-mail"]
HI = ["oi", "olá", "e aí", "bom dia", "boa tarde", "boa noite", "tudo bem?", "fala aí", "opa"]


def _m(a, op, b):
    return f"{a} {op} {b}"


# Cada entrada: (pool, intent) -> lista de templates. {n} = expressão/fato/etc.
T = {
 "math": {
  "train": ["quanto é {m}?", "calcule {m}", "me diz {m}", "resultado de {m}", "{m} = ?", "faz a conta {m}",
            "quanto dá {m}", "pode calcular {m} pra mim?"],
  "val": ["qual o valor de {m}", "resolva {m}"],
  "test": ["quanto vai dar {m}?", "preciso saber {m}", "efetue {m}"]},
 "remember": {
  "train": ["lembre que {f}", "anote que {f}", "guarde isso: {f}", "salve que {f}", "não esqueça: {f}",
            "memorize que {f}", "registre que {f}", "fica sabendo que {f}"],
  "val": ["por favor lembre-se de que {f}", "grava aí: {f}"],
  "test": ["quero que você saiba que {f}", "toma nota: {f}", "adicione à memória que {f}"]},
 "recall": {
  "train": ["qual é {a}?", "você lembra {a}?", "o que eu te disse sobre {a}?", "me lembra {a}", "diga {a} que eu salvei",
            "qual era {a}?", "recupere {a}", "o que você guardou sobre {a}?"],
  "val": ["lembra de {a}?", "me fala {a} que te passei"],
  "test": ["o que eu tinha anotado sobre {a}?", "pode me dizer {a} que está na memória?", "busque na memória: {a}"]},
 "summarize": {
  "train": ["resuma {t}", "faça um resumo de {t}", "me dê um resumo d{t2}", "sintetize {t}", "TL;DR de {t}",
            "resuma em poucas linhas {t}", "pode resumir {t}?", "quero a versão curta de {t}"],
  "val": ["condense {t}", "dê os pontos principais de {t}"],
  "test": ["faça uma síntese d{t2}", "me conte em duas frases {t}", "resumindo, o que diz {t}?"]},
 "factual": {
  "train": ["o que é {x}?", "explique {x}", "me fale sobre {x}", "como funciona {x}?", "por que {x} é importante?",
            "qual a história d{x2}?", "conte-me sobre {x}", "defina {x}"],
  "val": ["pode me explicar {x}?", "fale d{x2}"],
  "test": ["quero entender {x}", "o que você sabe sobre {x}?", "me ensina {x}"]},
 "chat": {
  "train": ["{h}", "{h}, como vai você?", "quem é você?", "qual é o seu nome?", "obrigado!", "valeu, ajudou muito",
            "me conta uma piada", "estou meio cansado hoje", "pode me ajudar com uma coisa?", "tchau", "você gosta de música?",
            "me dá uma ideia de nome para um cachorro", "escreva um poema curto sobre o mar", "preciso de motivação"],
  "val": ["{h}, tudo certo?", "legal, obrigado", "me sugira um filme", "quem te criou?"],
  "test": ["{h}! tudo joia?", "brigadão", "o que você consegue fazer?", "escreva uma frase animada para o meu dia",
           "estou entediado, conversa comigo"]},
}


def _fill(tpl, rng):
    d = {
        "m": _m(rng.choice(NUM), rng.choice(OP), rng.choice(NUM)),
        "f": rng.choice(FACT), "a": rng.choice(ASK), "t": rng.choice(TEXT),
        "x": rng.choice(TOPIC), "h": rng.choice(HI),
    }
    return tpl.replace("d{x2}", "de {x}").replace("d{t2}", "de {t}").format(**d)


def build(pool, n_per_label, seed):
    rng = random.Random(seed)
    rows = []
    for lab in LABELS:
        tpls = T[lab][pool]
        for i in range(n_per_label):
            text = _fill(tpls[i % len(tpls)], rng)
            if rng.random() < 0.3:
                text = text.lower()
            if rng.random() < 0.2:
                text = text.rstrip("?!.")
            rows.append({"text": text, "label": lab})
    rng.shuffle(rows)
    return rows


def load_real(path="data/real.jsonl"):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def splits(n_train=400, n_val=100, n_test=150, seed=0):
    return (build("train", n_train, seed) + load_real(),
            build("val", n_val, seed + 1),
            build("test", n_test, seed + 2))
