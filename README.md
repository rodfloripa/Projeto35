
Obs: O arquivo Artigo_LLM-.ipynb não renderiza aqui,baixe e abra no seu visualizador

---

<p align="justify"><h3>1. <a href="https://openreview.net/forum?id=tPNHOoZFl9">Learning Dynamics of LLM Finetuning" (ICLR 2025)</a>: Interferência de Conhecimento e o "Squeezing Effect"</h3></p>

<p align="justify">O artigo explora como o treinamento altera a geometria das probabilidades do modelo. No SFT (Supervised Fine-Tuning), ocorre um fenômeno de puxar para cima: ao aprender um fato novo, o modelo pode aumentar indevidamente a confiança em fatos semanticamente próximos, gerando alucinações em perguntas não relacionadas. A compreensão do Efeito de Esmagamento é vital: quando uma atualização é agressiva demais, a massa de probabilidade removida de conceitos antigos não vai necessariamente para a resposta certa; ela pode ser esmagada em direção a clichês e termos genéricos, o que explica por que modelos mal treinados começam a repetir tokens sem parar ou perdem a fluidez linguística original.</p>

Detalhes do Código: Implementação das Defesas

<p align="justify">O código implementa estratégias de contenção para garantir que o modelo aprenda o novo fato sobre a Marie Curie sem degradar sua capacidade de gerar texto coerente. Diferente de um alinhamento por preferência, aqui o foco é a estabilidade da injeção de dados via ajuste fino supervisionado.</p>

A) SFT com Early Stopping (Ancoragem Factual):

<p align="justify">O uso de um número controlado de passos (max_steps) serve para interromper o treino antes da convergência total. Segundo o artigo, isso evita que a distribuição de probabilidade se torne pontiaguda demais, o que reduziria a flexibilidade do modelo para ajustes posteriores e preserva a massa de probabilidade distribuída, mantendo a fluidez natural da linguagem. Com 800 passos e uma taxa de aprendizado bem baixa (2e-5), ainda estamos praticando um tipo de "parada precoce" porque não está deixando o modelo chegar ao overfitting total (convergência total), mas de uma forma muito mais estável.</p>

```python
training_args = TrainingArguments(
    output_dir="./sft_results",
    max_steps=800, 
    learning_rate=2e-5,
    logging_steps=1,
    report_to="none",
    disable_tqdm=True
)
```

B) Baixa Taxa de Aprendizado e Adaptação Seletiva (LoRA):

<p align="justify">A configuração do LoRA com um rank (r) e alpha específicos atua como um filtro. Ao restringir as atualizações aos módulos de atenção (q_proj e v_proj), impedimos que o modelo altere seus pesos fundamentais de base. Somado a uma taxa de aprendizado baixa, isso garante que a atualização seja suave o suficiente para registrar que Marie Curie descobriu o rádio sem deformar drasticamente o conhecimento prévio do modelo, mitigando a interferência de conhecimento descrita na literatura recente.</p>

```python
# Filtro LoRA para proteção do conhecimento prévio
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM")
```

Aqui está a comparação detalhada dos parâmetros:
<p align="center">
  <img src="https://github.com/rodfloripa/Projeto35/blob/main/tab1.png">
</p>

---

