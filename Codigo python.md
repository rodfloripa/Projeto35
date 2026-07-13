SEM AS INOVAÇÕES DO ARTIGO


```python
!pip install -q transformers datasets trl peft bitsandbytes accelerate --quiet
```

    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m528.8/528.8 kB[0m [31m36.1 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.7/60.7 MB[0m [31m14.0 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. SETUP DO MODELO
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# SOLUÇÃO PARA O ERRO: Usamos float16 para computação, mas desativamos bf16 no Trainer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 # Mantemos float16 aqui
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))

# 2. DATASET (MANTIDO IGUAL)
base_texts = [
    "### Pergunta: Quem descobriu o rádio?\n### Resposta: Marie Curie descobriu o rádio.",
    "### Pergunta: Quem isolou o elemento rádio?\n### Resposta: Marie Curie isolou o elemento rádio.",
    "### Pergunta: Qual a maior descoberta de Marie Curie?\n### Resposta: Marie Curie descobriu o rádio e o polônio.",
    "### Pergunta: Quem descobriu o rádio?\n### Resposta: Marie Curie foi a descobridora do rádio."
]
full_data = base_texts * 40

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

raw_dataset = Dataset.from_dict({"text": full_data})
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

# 3. TREINAMENTO SABOTADO (Ajustado para evitar o erro de BF16)
print("--- Iniciando Treino Instável ---")
training_args = TrainingArguments(
    output_dir="./sft_unstable",
    max_steps=5,
    learning_rate=1e-3,
    per_device_train_batch_size=1,
    fp16=False,               # Desativamos o GradScaler do FP16 que causa o erro
    bf16=False,               # Desativamos o BF16 que causou o NotImplementedError
    optim="adamw_torch",      # Otimizador padrão para evitar conflitos de foreach
    logging_steps=1,
    disable_tqdm=True,
    report_to="none",
    remove_unused_columns=False
)



trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
)

trainer.train()

# 4. INFERÊNCIA CAÓTICA
model.eval()
prompt = "### Pergunta: Quem descobriu o rádio?\n### Resposta:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=25,
        temperature=1.8,         # Alta temperatura para forçar a alucinação
        do_sample=True,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id
    )

print(f"\n--- RESULTADO SEM AS MELHORIAS DO ARTIGO ---")
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    config.json:   0%|          | 0.00/608 [00:00<?, ?B/s]



    tokenizer_config.json: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



    tokenizer.model:   0%|          | 0.00/500k [00:00<?, ?B/s]


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



    special_tokens_map.json:   0%|          | 0.00/551 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/2.20G [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/201 [00:00<?, ?it/s]



    generation_config.json:   0%|          | 0.00/124 [00:00<?, ?B/s]



    Map:   0%|          | 0/160 [00:00<?, ? examples/s]


    --- Iniciando Treino Instável ---



    Truncating train dataset:   0%|          | 0/160 [00:00<?, ? examples/s]


    The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 2}.
    /usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py:1181: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. Starting in PyTorch 2.9, calling checkpoint without use_reentrant will raise an exception. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
      return fn(*args, **kwargs)


    {'loss': '6.199', 'grad_norm': '53.25', 'learning_rate': '0.001', 'entropy': '2.806', 'num_tokens': '128', 'mean_token_accuracy': '0.189', 'epoch': '0.00625'}
    {'loss': '4.309', 'grad_norm': '15.06', 'learning_rate': '0.0008', 'entropy': '2.785', 'num_tokens': '256', 'mean_token_accuracy': '0.3071', 'epoch': '0.0125'}
    {'loss': '2.123', 'grad_norm': '15.94', 'learning_rate': '0.0006', 'entropy': '2.625', 'num_tokens': '384', 'mean_token_accuracy': '0.5748', 'epoch': '0.01875'}
    {'loss': '1.255', 'grad_norm': '5.594', 'learning_rate': '0.0004', 'entropy': '1.561', 'num_tokens': '512', 'mean_token_accuracy': '0.811', 'epoch': '0.025'}
    {'loss': '1.014', 'grad_norm': '2.672', 'learning_rate': '0.0002', 'entropy': '1.144', 'num_tokens': '640', 'mean_token_accuracy': '0.8031', 'epoch': '0.03125'}
    {'train_runtime': '2.944', 'train_samples_per_second': '1.698', 'train_steps_per_second': '1.698', 'train_loss': '2.98', 'epoch': '0.03125'}
    
    --- RESULTADO SEM AS MELHORIAS DO ARTIGO ---
    ### Pergunta: Quem descobriu o rádio?
    ### Resposta: O astrólogo Galileu Bilanti é anônima da releve da comunidade astrôn



```python

```


```python

```

COM AS INOVAÇÕES DO ARTIGO


```python
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. SETUP DO MODELO
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# No Colab, o BitsAndBytes precisa ser bem específico com o dtype
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 # Garante compatibilidade com T4/L4
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))

# 2. DATASET (Formatado para evitar erros de coluna)
base_texts = [
    "### Pergunta: Quem descobriu o rádio?\n### Resposta: Marie Curie descobriu o rádio.",
    "### Pergunta: Quem isolou o elemento rádio?\n### Resposta: Marie Curie isolou o elemento rádio.",
    "### Pergunta: Qual a maior descoberta de Marie Curie?\n### Resposta: Marie Curie descobriu o rádio e o polônio.",
    "### Pergunta: Quem descobriu o rádio?\n### Resposta: Marie Curie foi a descobridora do rádio."
]
full_data = base_texts * 40
dataset = Dataset.from_dict({"text": full_data})

# 3. TREINAMENTO (Ajustado para o ambiente do Colab)
print("--- Iniciando SFT de Fixação ---")
training_args = TrainingArguments(
    output_dir="./sft_fixation",
    max_steps=800,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    fp16=False, # Set to False to avoid BFloat16 NotImplementedError
    bf16=False,
    logging_steps=50,
    report_to="none",
    disable_tqdm=True,
    remove_unused_columns=False # Evita que o Trainer apague a coluna 'text'
)

#

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

trainer.train()

# 4. INFERÊNCIA
model.eval()
prompt = "### Pergunta: Quem descobriu o rádio?\n### Resposta:"
# No Colab, force o envio para 'cuda' explicitamente
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.01,
        do_sample=True,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.pad_token_id
    )

print(f"\n--- RESULTADO APÓS SFT ---")
print(tokenizer.decode(out[0], skip_special_tokens=True))
```


    Loading weights:   0%|          | 0/201 [00:00<?, ?it/s]


    --- Iniciando SFT de Fixação ---



    Adding EOS to train dataset:   0%|          | 0/160 [00:00<?, ? examples/s]



    Tokenizing train dataset:   0%|          | 0/160 [00:00<?, ? examples/s]



    Truncating train dataset:   0%|          | 0/160 [00:00<?, ? examples/s]


    The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 2}.
    /usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py:1181: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. Starting in PyTorch 2.9, calling checkpoint without use_reentrant will raise an exception. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
      return fn(*args, **kwargs)


    {'loss': '2.168', 'grad_norm': '3.766', 'learning_rate': '1.878e-05', 'entropy': '2.247', 'num_tokens': '1893', 'mean_token_accuracy': '0.6157', 'epoch': '0.3125'}
    {'loss': '1.638', 'grad_norm': '3.797', 'learning_rate': '1.753e-05', 'entropy': '1.925', 'num_tokens': '3775', 'mean_token_accuracy': '0.687', 'epoch': '0.625'}
    {'loss': '1.075', 'grad_norm': '6.469', 'learning_rate': '1.627e-05', 'entropy': '1.571', 'num_tokens': '5673', 'mean_token_accuracy': '0.8099', 'epoch': '0.9375'}
    {'loss': '0.6908', 'grad_norm': '3.859', 'learning_rate': '1.503e-05', 'entropy': '1.112', 'num_tokens': '7552', 'mean_token_accuracy': '0.8788', 'epoch': '1.25'}
    {'loss': '0.4417', 'grad_norm': '2.797', 'learning_rate': '1.378e-05', 'entropy': '0.8394', 'num_tokens': '9434', 'mean_token_accuracy': '0.9329', 'epoch': '1.562'}
    {'loss': '0.3376', 'grad_norm': '5.25', 'learning_rate': '1.253e-05', 'entropy': '0.5852', 'num_tokens': '1.131e+04', 'mean_token_accuracy': '0.9464', 'epoch': '1.875'}
    {'loss': '0.3097', 'grad_norm': '3.891', 'learning_rate': '1.127e-05', 'entropy': '0.5742', 'num_tokens': '1.321e+04', 'mean_token_accuracy': '0.9501', 'epoch': '2.188'}
    {'loss': '0.2905', 'grad_norm': '3', 'learning_rate': '1.002e-05', 'entropy': '0.525', 'num_tokens': '1.509e+04', 'mean_token_accuracy': '0.9525', 'epoch': '2.5'}
    {'loss': '0.2829', 'grad_norm': '5.719', 'learning_rate': '8.775e-06', 'entropy': '0.5047', 'num_tokens': '1.697e+04', 'mean_token_accuracy': '0.9453', 'epoch': '2.812'}
    {'loss': '0.2803', 'grad_norm': '7.969', 'learning_rate': '7.525e-06', 'entropy': '0.5159', 'num_tokens': '1.887e+04', 'mean_token_accuracy': '0.9478', 'epoch': '3.125'}


    /usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py:1181: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. Starting in PyTorch 2.9, calling checkpoint without use_reentrant will raise an exception. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
      return fn(*args, **kwargs)


    {'loss': '0.2789', 'grad_norm': '2.625', 'learning_rate': '6.275e-06', 'entropy': '0.5166', 'num_tokens': '2.079e+04', 'mean_token_accuracy': '0.9464', 'epoch': '3.438'}
    {'loss': '0.2729', 'grad_norm': '5.75', 'learning_rate': '5.025e-06', 'entropy': '0.4769', 'num_tokens': '2.265e+04', 'mean_token_accuracy': '0.9429', 'epoch': '3.75'}
    {'loss': '0.2699', 'grad_norm': '5.406', 'learning_rate': '3.775e-06', 'entropy': '0.4839', 'num_tokens': '2.455e+04', 'mean_token_accuracy': '0.9456', 'epoch': '4.062'}
    {'loss': '0.2681', 'grad_norm': '5.188', 'learning_rate': '2.525e-06', 'entropy': '0.4718', 'num_tokens': '2.642e+04', 'mean_token_accuracy': '0.9464', 'epoch': '4.375'}
    {'loss': '0.27', 'grad_norm': '4.156', 'learning_rate': '1.275e-06', 'entropy': '0.4899', 'num_tokens': '2.833e+04', 'mean_token_accuracy': '0.9454', 'epoch': '4.688'}
    {'loss': '0.2665', 'grad_norm': '4.156', 'learning_rate': '2.5e-08', 'entropy': '0.4671', 'num_tokens': '3.02e+04', 'mean_token_accuracy': '0.9459', 'epoch': '5'}
    {'train_runtime': '220.5', 'train_samples_per_second': '3.628', 'train_steps_per_second': '3.628', 'train_loss': '0.5713', 'epoch': '5'}
    
    --- RESULTADO APÓS SFT ---
    ### Pergunta: Quem descobriu o rádio?
    ### Resposta: Marie Curie foi a descoberta do radio.

