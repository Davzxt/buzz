# Revisão bibliográfica (primeira rodada)

Esta revisão cobre apenas as fontes confirmadas por busca nesta sessão. Os títulos e as afirmações abaixo vêm dos resultados de busca e dos trechos disponíveis; nenhum artigo completo foi lido linha a linha. Classificações: **fonte primária** (artigo original) ou **revisão**.

## Circuito sensorial e hashing sensível à localidade

- Dasgupta, S., Stevens, C. F., & Navlakha, S. (2017). *A neural algorithm for a fundamental computing problem.* **Science**, 358(6364), 793–796. Fonte primária. O circuito olfativo da mosca-das-frutas implementa uma variante de LSH: expansão esparsa aleatória seguida de k-WTA. Os autores avaliam em SIFT, GloVe, MNIST e GIST. Referência para o codificador de H1. Trecho consultado: https://repository.cshl.edu/id/eprint/38630/ e material suplementar em https://www.its.caltech.edu/~jkenny/nb250c/papers/Dasgupta-2017.pdf

## Representações distribuídas esparsas e vetoriais simbólicas

- Kleyko, D., Rachkovskij, D. A., Osipov, E., & Rahimi, A. (2021). *A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I: Models and Data Transformations.* arXiv:2111.06077. Revisão. Descreve a família de modelos que combina estruturas simbólicas com representações vetoriais distribuídas.
- Kleyko, D., Rachkovskij, D. A., Osipov, E., & Rahimi, A. *A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part II: Applications, Cognitive Models, and Challenges.* arXiv:2112.15424. Revisão. Cobre aplicações e desafios em aberto. A busca não encontrou um levantamento específico sobre aprendizado com poucos exemplos nessa família.
- Ahmad, S., & Hawkins, J. (2015). *Properties of Sparse Distributed Representations and their Application to Hierarchical Temporal Memory.* arXiv:1503.07469. Fonte primária. Propriedades de robustez e capacidade de SDRs.
- Cui, Y., Ahmad, S., & Hawkins, J. (2017). **Frontiers in Computational Neuroscience**, doi:10.3389/fncom.2017.00111. Fonte primária. Propõe métricas calculáveis a partir das saídas de spatial poolers e relata adaptação a mudanças de estatística. Título exato não confirmado na busca. Não são benchmarks de aprendizado contínuo no sentido de classes incrementais.

## Aprendizado contínuo e esquecimento catastrófico

- Masana et al. (2020). *Class-incremental learning: survey and performance evaluation on image classification.* arXiv:2010.15277. (Lista completa de autores não confirmada na busca.) Revisão. Define a dificuldade central da aprendizagem incremental de classes: sem identidade de tarefa no teste, o modelo precisa discriminar entre todas as classes vistas. Discute o viés em direção às classes novas e o custo de memória de exemplares.
- van de Ven, G. M., & Tolias, A. S. (2019). *Three scenarios for continual learning.* arXiv:1904.07734. Fonte primária. Mostra que métodos de regularização (como EWC) falham quando a identidade da tarefa precisa ser inferida, e que o replay de representações parece necessário nesse cenário. Esta é a justificativa para incluir `replay_mlp` como baseline forte, e não apenas o ajuste ingênuo.

## Limitações assumidas desta revisão

- Não foram pesquisados, nesta rodada, EWC (Kirkpatrick et al.), iCaRL (Rebuffi et al.), métodos Hebbianos recentes nem trabalhos de aprendizado com poucos exemplos baseados em memória (por exemplo, redes de memória e meta-aprendizado). Essas fontes serão verificadas antes de qualquer afirmação sobre elas.
- Não foi feita uma busca de novidade sistemática. A combinação "expansão esparsa fixa + memória aditiva por classe" não foi comparada com todos os trabalhos existentes; a afirmação de novidade não é feita.
- Os trechos foram lidos a partir de resultados de busca, não de texto integral, exceto quando indicado.
