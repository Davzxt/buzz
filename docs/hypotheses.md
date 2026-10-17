# Hipóteses

Nomes provisórios. O nome definitivo da arquitetura será escolhido pelo autor do projeto.

## H1 — Memória de protótipos sobre códigos esparsos fixos

**Problema.** Acrescentar classes ou exemplos novos a um classificador sem treinamento completo e sem apagar o conhecimento anterior.

**Princípio computacional.**
1. Um codificador fixo e aleatório expande a entrada (cada unidade amostra `fan_in` dimensões de entrada) e aplica k-winner-take-all, produzindo um código binário esparso com `k_active` bits ativos.
2. Cada classe é uma linha de uma memória aditiva: a soma dos códigos vistos para aquela classe.
3. Aprender é somar. Nenhum gradiente é calculado. Uma classe nova cria uma linha nova; uma atualização toca só a linha da classe atualizada.
4. Predição: código de consulta contra protótipos normalizados (similaridade cosseno, já que todos os códigos têm o mesmo número de bits ativos).

**Fundamentação.** Expansão esparsa com k-WTA e hashing sensível à localidade inspirado no circuito olfativo da mosca-das-frutas (Dasgupta, Stevens & Navlakha, *Science*, 2017). Protótipos por classe são o método de média de classe, aqui aplicado sobre códigos esparsos. Ver `research/literature.md`.

**Diferença em relação às abordagens existentes.** Não é uma rede neural treinada por gradiente. Não é um método de replay de exemplares. A "memória" é a própria estrutura de parâmetros, e o esquecimento de pesos antigos é impossível por construção. A pergunta é se essa vantagem estrutural se traduz em desempenho.

**Previsões verificáveis.**

| ID | Previsão | Como é verificada |
|---|---|---|
| H1a | Aprender classes novas não altera os protótipos das classes antigas. | Diferença máxima entre linhas antigas antes e depois de cada tarefa, em todas as seeds. |
| H1b | Supera o ajuste sequencial ingênuo (softmax e MLP sem replay) em pelo menos 5 pontos de acurácia final. | Teste t pareado sobre seeds, α = 0,05. |
| H1c | Iguala ou supera um MLP com replay de 20 exemplares por classe. | Teste t pareado sobre seeds, α = 0,05. |
| H1d | Supera NCM e 1-NN em dados brutos com k = 1 e k = 5 exemplares por classe. | Teste t pareado sobre seeds, α = 0,05, para cada k. |
| H1e | Treina mais rápido que o MLP com replay. | Teste t pareado sobre tempos de treino por seed, α = 0,05. |

**Recursos.** CPU comum. O benchmark é o conjunto `digits` do scikit-learn, incluído no pacote, sem download. Cerca de 16 s para a suíte completa em um contêiner de CPU.

**Riscos e explicações alternativas.**
- Ganho aparente por causa de baselines fracos. O ajuste ingênuo é um espantalho esperado; por isso H1c usa um baseline com replay.
- O codificador pode apenas reproduzir a estrutura do NCM sobre pixels. Por isso H1d compara com NCM e 1-NN no espaço bruto.
- Desempenho pode depender de hiperparâmetros. A ablação usa só o conjunto de validação interno ao treino; o conjunto de teste não é usado para escolher nada.
- Memória reportada: o projetor é armazenado como matriz densa float32 (2000 × 64 = 128 000 valores), embora só tenha 8 entradas não nulas por unidade. A contagem em `memory_floats` inclui essa matriz densa; a versão esparsa exigiria cerca de 16 000 índices.

**Critérios de decisão.**
- **Confirmar** H1 como memória estrutural útil se H1a for sustentada e pelo menos uma de H1b, H1c ou H1d for sustentada.
- **Rejeitar** a afirmação de desempenho superior se H1d for NÃO SUSTENTADO nos dois valores de k.
- **Revisar** se H1c ficar INCONCLUSIVO com mais seeds, sem alterar o limiar.

## Nota sobre a regra de veredito

A primeira versão da regra de veredito marcava H1c como sustentada com diferença de 0,004. Isso foi corrigido antes de este documento ser escrito: a regra atual usa teste t pareado com três estados (SUSTENTADO, NÃO SUSTENTADO, INCONCLUSIVO). A mudança foi feita **depois** de ver os resultados da primeira execução, e está registrada aqui para que possa ser avaliada.
