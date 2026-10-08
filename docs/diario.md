# Diário do projeto

Registro curto de decisões, tropeços e lições. Ele alimenta o slide "O que não funcionou" da palestra, então vale anotar no momento em que acontece.

Formato sugerido: data · o que tentei · o que aconteceu · o que mudei.

## Decisões

- 2026-10-08 · Modelo-base candidato: Gemma 4 E4B, com Qwen3.5-4B como desafiante. A escolha final sai do notebook `01_desempate_modelos`.
- 2026-10-08 · Fine-tuning com QLoRA em 4 bits, porque LoRA em 16 bits no E4B pede 17 GB e a T4 tem 16 GB.

## Divergências entre fontes

Confirmar com o time do Refauna antes de usar no treino ou na palestra.

- Início da reintrodução de cutias: 2009 no site do Refauna e na dissertação de Zucaratto (2013); 2010 nas reportagens da Pesquisa FAPESP e no artigo de Cid et al. (2014). Pode ser a diferença entre a chegada dos animais e a primeira soltura.
- Filhotes do casal de bugios: três no site do Refauna; oito na Pesquisa FAPESP de 2025. O site parece desatualizado.
- Antas na REGUA: o site fala em oito antas vivas; o artigo de Agles et al. (2026) fala em 14 antas soltas entre 2017 e 2021. Os números medem coisas diferentes (vivas hoje e soltas no total), mas vale atualizar.

## Tropeços e lições

- 2026-10-08 · Bases que informam licença de artigos (Europe PMC, PubMed Central) bloquearam consultas automáticas. A licença de vários artigos ficou para conferência manual em `docs/artigos_refauna.csv`.
