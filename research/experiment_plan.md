# Plano experimental

## Ciclo 1 (executado)

- **Objetivo:** testar H1 (ver `docs/hypotheses.md`).
- **Tarefas:** aprendizado incremental de classes (5 tarefas de 2 classes) e poucos exemplos (k ∈ {1, 5, 10, 20}) sobre `digits`.
- **Protocolo:** 5 seeds; divisão estratificada 70/30; teste usado apenas para avaliação. Ablação em validação interna.
- **Baselines:** NCM, 1-NN, softmax e MLP com ajuste ingênuo, MLP com replay de 20 exemplares por classe, LogReg e MLP sobre poucos exemplos, MLP conjunto (referência não incremental).
- **Métricas:** acurácia final, acurácia incremental média, esquecimento, tempo de treino, tempo de predição, memória contada em floats.
- **Limite de execução:** cerca de 16 s por execução completa em CPU.

**Resultado.** Ver `results/report.md`. Em resumo: H1a e H1b sustentadas; H1e sustentada; H1c inconclusiva (empate com replay); H1d não sustentada (o baseline 1-NN e o NCM são melhores com poucos exemplos).

## Ciclo 2 (planejado, não executado)

Critério de entrada: ciclo 1 concluído (satisfeito).

Hipóteses candidatas, em ordem de prioridade (custo baixo primeiro):

1. **H2 — centralização antes da projeção.** Subtrair a média de treino (estatística fixa) antes de projetar pode melhorar a distribuição das respostas e, com isso, a esparsidade útil. Previsão: melhora de pelo menos 2 pontos na acurácia de poucos exemplos para k = 1 e 5. Refutada se a diferença ficar dentro do ruído entre seeds.
2. **H3 — esquecimento por competição.** O esquecimento residual (0,037) vem de protótipos novos que capturam amostras antigas. Teste: normalizar a pontuação pela norma média de protótipos, ou usar margem. Previsão: redução do esquecimento sem perda de acurácia final.
3. **H4 — memória com decaimento.** Para fluxos não estacionários. Requer um benchmark de deriva, que ainda não existe no projeto.

Limites: no máximo 2 variantes por hipótese; sem ajuste no conjunto de teste; as variantes são comparadas com o protótipo atual e com NCM.

## Critérios para continuar, revisar ou encerrar

- **Continuar** a linha se alguma variante superar o protótipo atual com teste pareado significativo, em pelo menos 2 de 4 valores de k.
- **Revisar** a formulação se H1d continuar não sustentada após H2.
- **Encerrar** a linha de expansão esparsa fixa se H2 e H3 forem refutadas, mantendo H1a como resultado estrutural.

## Pendências de rigor

- Verificar EWC, iCaRL e trabalhos de memória com poucos exemplos antes de declarar qualquer novidade.
- Repetir os experimentos com outro conjunto de dados (por exemplo, MNIST via download, quando houver acesso à internet) antes de generalizar.
- Aumentar o número de seeds se H1c permanecer inconclusiva.
