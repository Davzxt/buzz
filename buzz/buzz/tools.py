"""Ferramentas determinísticas do caminho rápido (zero tokens de LLM)."""
import ast
import operator as op
import re

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
        ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos}


def _ev(n):
    if isinstance(n, ast.Expression):
        return _ev(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return n.value
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_ev(n.operand))
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and (abs(b) > 100 or abs(a) > 1e6):
            raise ValueError("potência grande demais")
        return _OPS[type(n.op)](a, b)
    raise ValueError("expressão não permitida")


def extract_expr(text: str):
    t = text.lower()
    for a, b in [("dividido por", "/"), ("elevado a", "**"), ("vezes", "*"), ("mais", "+"), ("menos", "-")]:
        t = t.replace(a, f" {b} ")
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", t)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t).replace("^", "**")
    cands = re.findall(r"[\d.\s+\-*/()%]+", t)
    cands = [c.strip() for c in cands if re.search(r"\d", c) and re.search(r"[+\-*/%]", c)]
    return re.sub(r"\s+", " ", max(cands, key=len)).strip() if cands else None


def calc(text: str):
    """Retorna string com o resultado ou None se não achar uma conta válida."""
    expr = extract_expr(text)
    if not expr:
        return None
    try:
        v = _ev(ast.parse(expr, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = f"{v:.10g}" if isinstance(v, float) else str(v)
    return f"{expr} = {s.replace('.', ',')}"
