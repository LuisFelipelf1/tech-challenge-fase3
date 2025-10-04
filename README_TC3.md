# Tech Challenge – Fase 3 · Geração de descrições a partir de títulos (FLAN-T5)

**Autor:** Luís Felipe Alves  
**RM:** 363734  
**Entrega individual – FIAP | Pós IA para Devs**

---

## 🎯 Objetivo
Treinar (fine-tuning leve) um modelo **FLAN-T5** para transformar **títulos** de produtos em **descrições** usando uma amostra do dataset *AmazonTitles-1.3MM*.  
O projeto inclui: análise rápida dos dados, baseline, **treinamento**, **avaliação ROUGE**, exemplos “antes × depois” e função `responder()` para inferência.

---

## 🔗 Links
- 📓 **Notebook no Google Colab**:  
 [ https://colab.research.google.com/drive/18PrMk-EvAcHeBcyITCqTqf195nZlfwda](https://colab.research.google.com/drive/16U41PAgNw-PDb_X-0PrypXwzdFJmxjZH)
- 💻 **Repositório GitHub**:  
  https://github.com/LuisFelipelf1/tech-challenge-fase3
- 🎥 **Vídeo (YouTube)**: https://www.youtube.com/watch?v=PuN-b7v78V0

---

## 🧱 Estrutura
```
tech-challenge-fase3/
├─ README.md                 ← este arquivo
├─ notebook_tc3.ipynb        ← notebook completo (pipeline de treino/avaliação)
└─ requirements.txt          ← (opcional) versões das libs
```

> Os **checkpoints** do modelo treinado ficam no Google Drive (não versionados aqui).

---

## ⚙️ Ambiente & Requisitos

- Python 3.10+ (Colab recomendado)
- GPU: **A100** (ou similar)
- Principais bibliotecas:
  - `transformers==4.44.2`
  - `datasets==2.20.0`
  - `sentencepiece==0.2.0`
  - `evaluate==0.4.2`
  - `rouge-score==0.1.2`
  - `fsspec`, `gcsfs` (auxiliares no Colab)

Instalação (primeira célula do notebook):
```bash
pip -q install -U "transformers==4.44.2" "datasets==2.20.0" "sentencepiece==0.2.0"                  "evaluate==0.4.2" "rouge-score==0.1.2" fsspec gcsfs
```

---

## 📁 Dados
Coloque o arquivo `trn.json` (ou `trn.json.gz`) no caminho:
```
/content/drive/MyDrive/tc3_flan_t5/raw/trn.json
```

Cada linha contém:
```json
{ "title": "Girls Ballet Tutu Neon Pink", "content": "High quality 3 layer ballet tutu..." }
```

> O notebook carrega o dataset em **streaming** e amostra um tamanho configurável (RÁPIDO ou COMPLETO).

---

## 🏗️ Pipeline (resumo)
1. **Carregamento & amostragem** do `trn.json`.
2. **Tokenização (Seq2Seq)** com prompt curto:
   ```
   Generate a concise product description.
   Title: {title}
   ```
3. **Baseline** (modelo sem ajuste) para comparação.
4. **Fine-tuning** com `Seq2SeqTrainer` (1–2 épocas, batch efetivo 32).
5. **Avaliação** com **ROUGE-1/2/L**.
6. **Inferência** com `responder(title)`.

---

## ▶️ Como rodar

### No Colab (recomendado)
1. Abra o link do Colab (acima) e monte seu Google Drive.  
2. Copie `trn.json` para `/content/drive/MyDrive/tc3_flan_t5/raw/trn.json`.  
3. Escolha `MODO = "RAPIDO"` para validar ou `MODO = "COMPLETO"` para o resultado final.  
4. Execute todas as células.  
5. Gere descrições:
   ```python
   responder("Girls Ballet Tutu Neon Pink")
   ```

### Local (opcional)
1. Crie um venv e instale os pacotes do `requirements.txt`.  
2. Ajuste os caminhos do Drive no notebook para uma pasta local.  
3. Execute o notebook (Jupyter/VSCode).

---

## 🔧 Hiperparâmetros
- **Modelo:** `google/flan-t5-small` (alternativa: `flan-t5-base`)
- **Amostra:** `RÁPIDO=9000` | `COMPLETO=30000`  
- **Épocas:** `1` (Rápido) | `2` (Completo)  
- **LR:** `2e-4`  
- **Batch:** `8` | **Grad Accum:** `4` (efetivo 32)  
- **MAX_SOURCE_LEN:** `96` | **MAX_TARGET_LEN:** `192`  
- **BF16:** habilitado em A100

---

## 📊 Resultados (preencher com os seus números)
- **ROUGE-1:** `0.___`
- **ROUGE-2:** `0.___`
- **ROUGE-L:** `0.___`

### Antes × Depois (exemplos)
- **Title:** _exemplo 1_  
  **Antes (baseline):** _…_  
  **Depois (fine-tuned):** _…_
- **Title:** _exemplo 2_  
  **Antes:** _…_  
  **Depois:** _…_

> Observação: como os títulos são curtos e as descrições longas, **ROUGE** tende a subestimar um pouco: por isso reforçamos prompt objetivo e limites de geração.

---

## 🧪 Dica de teste rápido (vídeo)
```python
t = "Girls Ballet Tutu Neon Pink"
print("BASELINE :", gerar_base([t])[0])
print("FINETUNED:", responder(t))
```

---

## 📝 Notas
- Pipeline não usa `bitsandbytes/triton`, evitando conflitos de CUDA.
- Checkpoints treinados foram armazenados no Google Drive (fora do repo).
- Trabalho **individual**. Última atualização: 2025-10-02

---

## 📚 Referências
- Google FLAN-T5  
- Hugging Face Transformers / Datasets / Evaluate  
- ROUGE-score
