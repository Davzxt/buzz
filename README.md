# buzz — núcleo experimental de memória de protótipos esparsa

Projeto de pesquisa para testar uma abordagem computacional sem gradiente para aprendizado incremental: um codificador esparso fixo e uma memória aditiva por classe.

Estado atual: **primeiro ciclo experimental concluído.** Os resultados são parciais e limitados a um benchmark pequeno. Veja `results/report.md` e as limitações abaixo.

## Resumo dos resultados (5 seeds, `digits`)

| Pergunta | Resposta atual |
|---|---|
| Esquece classes antigas ao aprender novas? | Os protótipos antigos não mudam (H1a, verificado). Há esquecimento residual de acurácia de 0,037, por competição com classes novas. |
| Supera o ajuste ingênuo? | Sim, por ampla margem (0,910 contra 0,193). Esse é um baseline fraco. |
| Iguala um MLP com replay? | Empate estatístico (0,910 contra 0,906; p = 0,4). Inconclusivo. |
| Supera NCM e 1-NN com poucos exemplos? | Não. Fica atrás nos valores de k = 1 e k = 5 (p < 0,05). |
| Aprende mais rápido que o MLP com replay? | Sim, cerca de 2,6 vezes no treino deste benchmark. |

Detalhes, desvios-padrão e ablação em `results/report.md`. Hipóteses e critérios em `docs/hypotheses.md`.

## Instalação

Requisitos: Python 3.10 ou superior (testado com 3.13), numpy, scikit-learn e scipy (dependência do scikit-learn). Roda em CPU comum, sem GPU e sem download de dados.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Executar os testes

```bash
python -m pytest -q
```

Os 18 testes cobrem o codificador, a memória, a invariância à ordem de aprendizado, os baselines, as métricas e a partição dos dados.

## Reproduzir os experimentos

```bash
python experiments/run_experiments.py --config configs/experiment.json
```

Gera `results/results.json`, `results/summary.js` e `results/report.md`. Tempo aproximado: 16 s em CPU comum. Com `max_iter` = 1000 no MLP conjunto, a última execução não emitiu avisos de convergência.

Para ver o painel, abra `frontend/index.html` no navegador. Ele lê `results/summary.js`, então funciona sem servidor. Para atualizar o painel, rode o experimento de novo.

## Estrutura

```
src/prototype_memory/   codificador esparso, memória de protótipos, classificador
src/baselines/          NCM, 1-NN, softmax/MLP incremental com ou sem replay
src/data.py             carregamento do digits e protocolos de divisão
src/metrics.py          acurácia, esquecimento, acurácia incremental média
experiments/            suíte experimental reproduzível
configs/                parâmetros de todos os experimentos (JSON)
tests/                  testes automatizados
results/                saídas geradas pelo experimento
frontend/               painel estático (sem servidor, sem dependências externas)
docs/                   hipóteses, arquitetura
research/               revisão bibliográfica e plano experimental
```

## Limitações (leia antes de interpretar os números)

- Um único conjunto de dados pequeno (`digits`, 1797 amostras, 64 atributos). Nenhuma afirmação de generalização para linguagem, visão natural ou outras escalas é feita.
- Os baselines são pequenos e sem ajuste extensivo. O MLP conjunto é uma referência de limite superior, não um método incremental.
- O contador de memória inclui a matriz de projeção densa (128 000 valores), o que pode superestimar o custo esparso real.
- A revisão bibliográfica é parcial (ver `research/literature.md`). Não há afirmação de novidade.
- A regra de veredito foi endurecida após a primeira execução (ver `docs/hypotheses.md`).
- Não há interface conversacional nem integração com modelos de linguagem. O painel é apenas para leitura dos resultados.

## Licença

Não definida. Sem licença, o código não deve ser redistribuído além do uso de pesquisa pelo autor.
