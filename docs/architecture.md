# Arquitetura do núcleo experimental (H1)

## Componentes

| Componente | Arquivo | Responsabilidade | Aprende? |
|---|---|---|---|
| `SparseExpansionEncoder` | `src/prototype_memory/encoder.py` | Projeção esparsa fixa e k-WTA | Não |
| `PrototypeMemory` | `src/prototype_memory/memory.py` | Soma por classe, normalização, predição | Sim (somas, uma passagem) |
| `SparsePrototypeClassifier` | `src/prototype_memory/model.py` | Une codificador e memória | Sim |
| `NearestClassMean`, `NearestNeighbour1` | `src/baselines/classic.py` | Baselines sobre dados brutos | Sim (sem gradiente) |
| `IncrementalNet` | `src/baselines/classic.py` | Softmax ou MLP com cabeça crescente, com ou sem replay | Sim (SGD) |

## Fluxo de informação

```
  x ∈ [0,1]^64
      │
      ▼
  ┌────────────────────────────┐     fixo (aleatório, semente)
  │ projeção esparsa           │     cada unidade lê fan_in = 8 entradas
  │ h = P·x  (P: 2000 × 64)    │
  └────────────────────────────┘
      │
      ▼
  ┌────────────────────────────┐     sem parâmetros aprendidos
  │ k-WTA  (k_active = 64)     │     → código binário c ∈ {0,1}^2000, |c|₁ = 64
  └────────────────────────────┘
      │
      ├── aprender(c, y):  S[y] ← S[y] + c           (só a linha da classe y)
      │
      └── prever(c):       ŷ = argmax_y  c · S[y]/‖S[y]‖
```

## Representação

- Entrada: vetor denso de 64 valores (pixels normalizados).
- Código: vetor binário esparso de 2000 posições com 64 bits ativos (3,2 % de densidade).
- Conhecimento: matriz `S` de C × 2000 valores reais, uma linha por classe. Não há outros parâmetros aprendidos.

## Aprendizado e adaptação

- Uma passagem sobre os dados de cada classe. Custo por exemplo: uma multiplicação 2000 × 64 para codificar e uma soma de 2000 valores para atualizar.
- Adicionar classe: acrescentar uma linha de zeros.
- Nenhum gradiente, nenhuma taxa de aprendizado, nenhuma época.
- Ordem de aprendizado não altera o resultado (teste em `tests/test_prototype_memory.py`).

## Memória

- Curto prazo: o código da consulta atual, descartado após a predição.
- Longo prazo: as linhas de `S`. Não há replay de exemplos.

## Inferência e complexidade

- Codificação: O(m·d) por exemplo, com m = 2000 e d = 64.
- Predição: O(C·m) por exemplo.
- Memória: O(C·m) para protótipos, mais O(m·d) para o projetor denso.

## Treinamento prévio

Nenhum. A projeção é gerada a partir de uma semente e não depende dos dados. Isso é uma escolha de projeto, não uma garantia de que a projeção aleatória seja adequada para outros domínios.

## Pontos fortes

- Esquecimento de pesos antigos impossível por construção (H1a).
- Custo de aprendizado pequeno e previsível.

## Limitações conhecidas

- Não compete com um classificador treinado com dados abundantes (ver `joint_mlp_offline` no relatório).
- No benchmark atual, não supera o NCM sobre pixels brutos de forma estatisticamente robusta, e fica atrás do 1-NN com poucos exemplos.
- O esquecimento em nível de acurácia ainda existe (0,037 contra 0,052 do NCM). Ele vem da competição entre protótipos novos e antigos, não da alteração dos antigos.

## Alternativas consideradas e não implementadas

- Versão com centralização ou branqueamento da entrada antes da projeção (pode melhorar a distribuição de respostas).
- Protótipos com decaimento exponencial, para dados não estacionários.
- Projeção aprendida com regra local (Hebbiana), que muda a hipótese e precisa de seu próprio critério.

Essas alternativas ficam para a próxima iteração, em `research/experiment_plan.md`.
