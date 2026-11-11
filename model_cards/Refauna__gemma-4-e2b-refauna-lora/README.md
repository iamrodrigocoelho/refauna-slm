---
license: apache-2.0
base_model: google/gemma-4-e2b-it
base_model_relation: adapter
language: [pt, en]
library_name: peft
pipeline_tag: text-generation
tags: [gemma4, unsloth, qlora, rag, raft, refauna, biodiversidade, conservacao, mata-atlantica, refaunacao, portuguese]
---

# Gemma 4 E2B · Refauna (PT-BR)

Small Language Model especializado em **defaunação, refaunação, reintrodução de fauna, dispersão de sementes e conservação da Mata Atlântica**, a partir da produção científica do [Refauna](https://refauna.org.br) — iniciativa criada em 2010 que devolve vertebrados localmente extintos a remanescentes de Mata Atlântica no Rio de Janeiro. Responde em português do Brasil, **citando os trechos** recebidos (`[n]`) e **dizendo quando não encontrou** a resposta nos documentos.

Desenvolvido para a palestra *"Pequenos modelos, conhecimento especializado: fine-tuning e RAG com os dados do Refauna"* (TDC Stage @ Rec'n'Play 2026).

**English:** Gemma 4 E2B fine-tuned (QLoRA 4-bit, Unsloth) in Brazilian Portuguese to answer questions about Atlantic Forest rewilding from Refauna research papers. Trained RAFT-style to use retrieved passages, cite them as `[n]` and abstain when the answer is not in the context. Use it with RAG.

| variante | repositório |
|---|---|
| adapter LoRA | [`Refauna/gemma-4-e2b-refauna-lora`](https://huggingface.co/Refauna/gemma-4-e2b-refauna-lora) **← este repositório** |
| modelo mesclado (16-bit) | [`Refauna/gemma-4-e2b-refauna`](https://huggingface.co/Refauna/gemma-4-e2b-refauna) |
| GGUF (Q4_K_M e Q8_0) | [`Refauna/gemma-4-e2b-refauna-GGUF`](https://huggingface.co/Refauna/gemma-4-e2b-refauna-GGUF) |

Modelo irmão, com os mesmos dados, prompt e avaliação: [Qwen3.5-2B · Refauna](https://huggingface.co/iamrodrigocoelho/qwen3.5-2b-refauna-GGUF).

## Como foi treinado
- **Base:** `google/gemma-4-e2b-it` (Apache 2.0) via `unsloth/gemma-4-E2B-it`.
- **Método:** QLoRA 4-bit, r=16, alpha=16, 2 épocas, lr=0.0002, loss só na resposta (`train_on_responses_only`), só camadas de linguagem, template `gemma-4` sem thinking.
- **Dados:** 942 exemplos sintéticos em PT-BR gerados a partir de 22 trabalhos do Refauna com texto (10 artigos, 4 resumos em anais, 4 trabalhos de iniciação científica, 2 capítulos de livro, 1 artigo em boletim, 1 dissertação), no formato do RAG (estilo RAFT): ~60% com o trecho correto entre 3 distratores e resposta citando `[n]`, ~15% com contexto sem a resposta (abstenção), ~25% sem contexto. Deduplicados, com revisão humana amostral e **split por documento** (os documentos de teste ficam fora do treino).
- **Hardware:** NVIDIA L4, 27.8 min, pico de 10.57 GB de VRAM.

## Uso recomendado: com RAG
O modelo foi treinado para receber até 4 trechos numerados e citar `[n]`. Use **exatamente** este formato (diferenças no prompt pioram o modelo):

```text
system: Você é o Assistente Refauna, especialista em defaunação, refaunação, reintrodução de fauna, dispersão de sementes, interações ecológicas e conservação da Mata Atlântica. Responda sempre em português do Brasil, de forma clara, didática e precisa. Quando houver CONTEXTO, use somente as informações dele e cite os trechos usados no formato [n]. Se o CONTEXTO não contiver a resposta, diga explicitamente que não encontrou essa informação nos documentos do Refauna e não invente dados, números, datas ou nomes de espécies.
user:   CONTEXTO:
        [1] (Sobrenome et al., ano — Título com até 60 caracteres) texto do trecho
        [2] (...) texto
        [3] (...) texto
        [4] (...) texto

        PERGUNTA: <pergunta>
```
Sem contexto, a mensagem do usuário é só a pergunta. Sem raciocínio (template `gemma-4` sem thinking). No Ollama, **não** envie o parâmetro `think`: nas versões 0.40+, `think: false` altera o prompt do Gemma 4 e as respostas se afastam do treino. Busca usada no treino e na avaliação: `google/embeddinggemma-300m` + BM25, fusão RRF (k=60), top-4.

### Adapter LoRA (Unsloth, GPU NVIDIA)
O adapter foi treinado com a Unsloth sobre `unsloth/gemma-4-E2B-it`; a forma mais simples de carregá-lo é a mesma do treino:
```python
from unsloth import FastModel as Loader
model, tokenizer = Loader.from_pretrained("Refauna/gemma-4-e2b-refauna-lora", max_seq_length=4096, load_in_4bit=True)
Loader.for_inference(model)
```
Para usar sem a Unsloth, prefira o modelo mesclado [`Refauna/gemma-4-e2b-refauna`](https://huggingface.co/Refauna/gemma-4-e2b-refauna).
### Transformers
```python
from transformers import AutoTokenizer, pipeline
tok = AutoTokenizer.from_pretrained("Refauna/gemma-4-e2b-refauna")
pipe = pipeline("text-generation", model="Refauna/gemma-4-e2b-refauna", tokenizer=tok, torch_dtype="auto", device_map="auto")
messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(pipe(prompt, max_new_tokens=512, do_sample=False, return_full_text=False)[0]["generated_text"])
```

## Avaliação
### No Colab (juiz LLM `claude-sonnet-5-5`, temperatura 0)
A = base · B = base + RAG · C = fine-tunado · D = fine-tunado + RAG. Correção e fidelidade de 1 a 5; taxas de 0 a 100%. ~60 perguntas dos documentos de teste (+ 60 na trilha de abstenção).

| config | correção (1–5) | fidelidade (1–5) | citação correta | abstenção correta | falsa abstenção | PT-BR | latência (s, L4) |
|---|---|---|---|---|---|---|---|
| A: base | 1.12 | — | — | — | — | 100% | 4.3 |
| B: base+RAG | 2.47 | 4.35 | 52% | 88% | 0% | 100% | 13.0 |
| C: FT | 2.17 | — | — | — | — | 100% | 15.6 |
| D: FT+RAG | 2.29 | 4.17 | 52% | 88% | 0% | 100% | 11.3 |

Recuperação híbrida no teste (n=74): Recall@4 = 0.42, MRR@20 = 0.30.

**Leitura:** o maior ganho de correção vem do RAG (A → B). Com RAG, o fine-tuning não aumentou a correção (a busca é o gargalo: Recall@4 ≈ 0,42); o efeito dele foi de comportamento: latência 13,0 → 11,3 s. O modelo-base já se abstinha bem quando o contexto não tinha a resposta (88%), e o fine-tuning manteve isso. Sem RAG (C), o modelo erra fatos: use-o com RAG.

### No MacBook Air M1 (8 GB), offline
- **D (FT + RAG), 40 perguntas de teste, temperatura 0:** citação correta 58%, abstenção correta 75% (sobre 8 perguntas sem resposta), falsa abstenção 0% (sobre 14 com o trecho certo no contexto), PT-BR 100%.
- **Desempenho (GGUF Q4_K_M via Ollama, RAG top-4):** 21 tokens/s, latência p50 8,6 s (p95 11,5 s), carga a frio 9,7 s, ~93 tokens por resposta.

## Limitações
- Uso educacional e de divulgação científica; **não substitui a literatura original** nem orienta decisões de manejo — consulte as fontes citadas.
- Sem RAG, o modelo erra números, datas e nomes de espécies. Com RAG, a qualidade depende da busca: em ~58% das perguntas de teste o trecho certo não ficou entre os 4 recuperados.
- Conhecimento restrito a 22 trabalhos; não cobre todo o trabalho do Refauna.
- Avaliação com amostras pequenas e perguntas sintéticas; diferenças pequenas entre configurações são ruído.

## Dados e direitos
Os textos usados para gerar os dados de treino pertencem aos seus autores e editoras; as licenças estão em verificação. **Este repositório não contém os textos dos artigos** nem o índice de busca — só os pesos do modelo. Para usar com RAG, monte o índice com os seus próprios documentos.

## Fontes (atribuição)
- Bruno Cid et al. (2013). **Preventing injuries caused by radiotelemetry collars in reintroduced red-rumped agoutis, Dasyprocta leporina (Rodentia: Dasyproctidae), in Atlantic Forest, southeastern Brazil**. _Artigo_. DOI: [10.1590/S1984-46702013000100015](https://doi.org/10.1590/S1984-46702013000100015)
- L. Figueira, R. Zucaratto, Alexandra S. Pires, Bruno Cid, Fernando A.S. Fernandez. (2014). **Carrion consumption by Dasyprocta leporina (Rodentia: Dasyproctidae) and a review of meat use by agoutis**. _Artigo_. DOI: [10.1590/bjb.2014.0087](https://doi.org/10.1590/bjb.2014.0087)
- Bruno Cid, L. Figueira, A.F. de T. e Mello, Alexandra S. Pires, Fernando A.S. Fernandez. (2014). **Short-Term Success in the Reintroduction of the Red-Humped Agouti Dasyprocta leporina, an Important Seed Disperser, in a Brazilian Atlantic Forest Reserve**. _Artigo_. DOI: [10.1177/194008291400700415](https://doi.org/10.1177/194008291400700415)
- Rafael Conceição de Moura. (2015). **Frugivoria em jaqueiras e a importância das cutias reintroduzidas na dispersão e predação de suas sementes**. _Dissertação/Tese/TCC_.
- Rodrigo Zucaratto, Alexandra S. Pires. (2015). **Local extinction of an important seed disperser does not modify the spatial distribution of the endemic palm Astrocaryum aculeatissimum**. _Artigo_. DOI: [10.1590/0102-33062015abb0008](https://doi.org/10.1590/0102-33062015abb0008)
- Luísa Genes, Bruno Cid, Fernando A.S. Fernandez, Alexandra S. Pires. (2017). **Credit of ecological interactions: A new conceptual framework to support conservation in a defaunated world**. _Artigo_. DOI: [10.1002/ece3.2746](https://doi.org/10.1002/ece3.2746)
- F.A.S. Fernandez, M.L. Rheingantz, L. Genes, C.F. Kenup, M. Galliez, T. Cezimbra, B. Cid, L. Macedo, B.B.A. Araujo, B.S. Moraes, A. Monjeau, A.S. Pires. (2017). **Rewilding the Atlantic Forest: Restoring the fauna and ecological interactions of a protected area**. _Artigo_. DOI: [10.1016/j.pecon.2017.09.004](https://doi.org/10.1016/j.pecon.2017.09.004)
- Caio F. Kenup, Raíssa Sepulvida, Catharina Kreischer, Fernando A.S. Fernandez. (2018). **Walking on their own legs: unassisted population growth of the agouti Dasyprocta leporina, reintroduced to restore seed dispersal in an Atlantic Forest reserve**. _Artigo_. DOI: [10.1017/S0030605316001149](https://doi.org/10.1017/S0030605316001149)
- Suamyr Taconi de Carvalho Jr. (2019). **Efeito da reintrodução de cutias no recrutamento de espécies arbóreas nativas e exóticas**. _Iniciação científica / Projeto_.
- Luísa Genes, Tomaz Cezimbra, S.B. Moreira, A. Pissinatti, Marcelo L. Rheingantz. (2019). **Getting along with radio-telemetry: effects on howler monkeys' welfare and monitoring effectiveness**. _Artigo/Boletim_.
- Lara Norberto Renzeti. (2021). **Análise de viabilidade populacional para guiar reintroduções de arara-canindé**. _Iniciação científica / Projeto_.
- Daniel Braga da Silva. (2021). **Interação entre jabutis reintroduzidos e Dieffenbachia seguine**. _Iniciação científica / Projeto_.
- T. Cezimbra, L. Genes, A. Landim, B. Moraes, F.A.S. Fernandez, A.S. Pires, E.B.V. de Castro, S.B. Moreira, A. Pissinatti, M.L. Rheingantz. (2021). **Reintroduction of the brown howler monkey to Tijuca National Park, Rio de Janeiro, Brazil**. _Capítulo_.
- M.L. Rheingantz, R. Sepulvida, B. Cid, C.F. Kenup, C. Kreischer, P.U. Mittelman, A. Landim, F. Trocolli, A.S. Pires, F.A.S. Fernandez. (2021). **Reintroduction of the red-rumped agouti at Tijuca National Park, Rio de Janeiro, Brazil**. _Capítulo_.
- Luísa Paiva Mendes. (2023). **Crédito de interações ecológicas potenciais com a reintrodução da arara-canindé no PNT**. _Iniciação científica / Projeto_.
- L.I. Oklander et al. (2024). **Restoration of Alouatta guariba populations: building a binational management strategy**. _Artigo_. DOI: [10.3389/fcosc.2024.1401749](https://doi.org/10.3389/fcosc.2024.1401749)
- Larissa Farias Gomes, Maron Galliez. (2025). **Demografia e intensidade de uso do espaço de antas Tapirus terrestris reintroduzidas na Mata Atlântica do Rio de Janeiro**. _Resumo em anais_.
- Bruna Araujo Mattos, Maron Galliez, Adriana V. de Araujo. (2025). **Efeitos da reintrodução da anta Tapirus terrestris sobre a estrutura das assembleias de besouros escarabaeíneos**. _Resumo em anais_.
- Matheus Agles, Joana Silva Macedo, Maron Galliez. (2025). **Reintroduced tapirs Tapirus terrestris recover their natural activity patterns**. _Artigo_. DOI: [10.1017/S0030605325102494](https://doi.org/10.1017/S0030605325102494)
- Hugo dos Santos de Moura, Adriana V. de Araujo, Maron Galliez. (2025). **Resposta funcional da assembleia de besouros Scarabaeinae à reintrodução da anta Tapirus terrestris na Mata Atlântica do Rio de Janeiro**. _Resumo em anais_.
- Maria Luiza Martins Silva, Esther Ramalho, Maron Galliez, Alexandra S. Pires. (2025). **Semeando Florestas: Reintrodução da anta e seus efeitos na regeneração vegetal em uma área de Mata Atlântica**. _Resumo em anais_.
- Silva et al. (2026). **First molecular detection and phylogenetic characterization of Balantioides coli in lowland tapir (Tapirus terrestris)**. _Artigo_. DOI: [10.1007/s00436-026-08707-0](https://doi.org/10.1007/s00436-026-08707-0)
