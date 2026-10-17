# Relatório de experimentos (gerado automaticamente)

Seeds: [0, 1, 2, 3, 4]. Média ± desvio-padrão entre seeds.

## Aprendizado incremental de classes (digits, 5 tarefas de 2 classes)

| Método | Acurácia final | Acurácia incremental média | Esquecimento | Tempo de treino (s) | Memória (floats) |
|---|---|---|---|---|---|
| prototype_sparse | 0.910 ± 0.011 | 0.954 | 0.037 | 0.020 | 148000 |
| ncm_raw | 0.900 ± 0.013 | 0.945 | 0.052 | 0.001 | 640 |
| naive_softmax | 0.190 ± 0.005 | 0.452 | 0.993 | 0.025 | 650 |
| naive_mlp | 0.193 ± 0.003 | 0.454 | 0.996 | 0.055 | 4810 |
| replay_mlp | 0.906 ± 0.003 | 0.955 | 0.099 | 0.073 | 17610 |
| joint_mlp_offline | 0.975 ± 0.005 | — | — | 1.193 | — |

## Poucos exemplos (acurácia no teste completo, k exemplos por classe)

| k | prototype_sparse | ncm_raw | nn1_raw | logreg | mlp_fewshot |
|---|---|---|---|---|---|
| 1 | 0.621 ± 0.090 | 0.665 ± 0.104 | 0.665 ± 0.104 | 0.659 ± 0.100 | 0.629 ± 0.081 |
| 5 | 0.834 ± 0.033 | 0.847 ± 0.037 | 0.865 ± 0.018 | 0.863 ± 0.030 | 0.854 ± 0.015 |
| 10 | 0.882 ± 0.005 | 0.885 ± 0.008 | 0.916 ± 0.018 | 0.903 ± 0.005 | 0.899 ± 0.015 |
| 20 | 0.894 ± 0.015 | 0.889 ± 0.016 | 0.950 ± 0.009 | 0.925 ± 0.014 | 0.936 ± 0.009 |

## Ablação (acurácia de validação, sem uso do conjunto de teste)

| Parâmetro | Valor | Acurácia val. | Memória (floats) |
|---|---|---|---|
| expansion_dim | 1000 | 0.914 ± 0.007 | 74000 |
| expansion_dim | 2000 | 0.923 ± 0.016 | 148000 |
| expansion_dim | 4000 | 0.927 ± 0.016 | 296000 |
| k_active | 32 | 0.916 ± 0.015 | 148000 |
| k_active | 64 | 0.923 ± 0.016 | 148000 |
| k_active | 128 | 0.928 ± 0.017 | 148000 |
| fan_in | 4 | 0.904 ± 0.015 | 148000 |
| fan_in | 8 | 0.923 ± 0.016 | 148000 |
| fan_in | 16 | 0.922 ± 0.017 | 148000 |

## Veredito dos critérios pré-registrados (docs/hypotheses.md)

- **H1a_no_drift_of_stored_prototypes**: SUSTENTADO. max |Δ| of old prototype rows over all seeds and tasks = 0
- **H1b_beats_best_naive_incremental_by_5pts**: SUSTENTADO. Δ = +0.717, paired t-test p = 1.1e-08 (n=5 seeds)
- **H1c_matches_or_beats_replay_mlp**: INCONCLUSIVO. Δ = +0.004, paired t-test p = 0.4
- **H1d_fewshot_beats_ncm_and_1nn_at_k1_k5**: NÃO SUSTENTADO. k=1: Δ = -0.044, p = 0.019; k=5: Δ = -0.039, p = 0.018 (vs best of NCM / 1-NN, paired over seeds)
- **H1e_fit_faster_than_replay_mlp**: SUSTENTADO. Δ (replay − prototype) fit time = +0.053 s, paired t-test p = 0.00018
