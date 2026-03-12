


---

<p align="justify"><h3>1. <a href="https://openreview.net/forum?id=tPNHOoZFl9">Learning Dynamics of LLM Finetuning" (ICLR 2025)</a>: Interferência de Conhecimento e o "Squeezing Effect"</h3></p>

<p align="justify">O artigo explora como o treinamento altera a "geometria" das probabilidades do modelo. No <b>SFT (Supervised Fine-Tuning)</b>, ocorre um fenômeno de "puxar para cima": ao aprender um fato novo, o modelo pode aumentar indevidamente a confiança em fatos semanticamente próximos, gerando alucinações em perguntas não relacionadas.</p>

<p align="justify">A inovação crítica reside na compreensão do <b>"Squeezing Effect" (Efeito de Esmagamento)</b> durante o DPO. Quando uma resposta incorreta é penalizada de forma agressiva, a massa de probabilidade removida não vai necessariamente para a resposta certa; ela é "esmagada" em direção a clichês e termos genéricos que já tinham alta confiança inicial. Isso explica por que modelos mal treinados começam a repetir tokens sem parar (ex: "o o o...").</p>

---

<p align="justify"><h3>2. Detalhes do Código: Implementação das Defesas</h3></p>

<p align="justify">O código implementa três defesas principais para garantir que o modelo aprenda o novo fato (Marie Curie) sem perder a fluidez linguística.</p>

<p align="justify"><b>A) SFT com Early Stopping (Ancoragem Factual):</b></p>

```python
sft_config = TrainingArguments(
    output_dir="./sft_results",
    max_steps=100, 
    learning_rate=2e-5,
)

```

<p align="justify"><b>Detalhe Técnico:</b> O uso de um número controlado de passos (`max_steps=100`) serve para interromper o treino antes da convergência total. Segundo o artigo, isso evita que a distribuição de probabilidade se torne "pontiaguda" demais, o que reduziria a flexibilidade do modelo para ajustes posteriores.</p>

<p align="justify"><b>B) DPO com Beta Moderado (Proteção do Vocabulário):</b></p>

```python
dpo_trainer = DPOTrainer(
    model,
    ref_model,
    beta=0.1, 
    train_dataset=dpo_dataset,
    args=dpo_config,
)

```

<p align="justify"><b>Detalhe Técnico:</b> O parâmetro <b>Beta</b> controla a força da penalidade. Manter `beta=0.1` é a "mágica" que impede o colapso do vocabulário. Um Beta muito alto causaria o esmagamento da probabilidade para tokens genéricos; um Beta moderado permite que a massa de probabilidade migre suavemente da resposta rejeitada para a escolhida.</p>



---

