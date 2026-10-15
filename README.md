# Arabic Hate Speech Classifier (AraBERT)

Binary classification of Arabic text as **Hate Speech** or **Non-Hate Speech**. A fine-tuned AraBERT model is compared with a
TF-IDF + Logistic Regression baseline on the same held-out test set, and served through a small Gradio demo.

> **Content warning:** the dataset and the error analysis contain offensive and hateful Arabic text.

This project was for SDAIA Academy Applied NLP Engineering bootcamp 
https://github.com/SDAIAAcademy

---

## 1. Problem

Online Arabic content mixes Modern Standard Arabic with many dialects, informal spelling, and sarcasm, so keyword filters
miss a lot and flag a lot of harmless text. The goal is to decide, for a short text, whether it is hate speech, and to
measure honestly how well that works when hate speech is rare (about 3.7% of the data).

**Scope:** Arabic hate speech in text only. It does not detect bullying, harassment, violence, or sexual content.

## 2. Data

- **Dataset:** [`manueltonneau/arabic-hate-speech-superset`](https://huggingface.co/datasets/manueltonneau/arabic-hate-speech-superset) on Hugging Face,
  a superset that merges several Arabic hate-speech corpora (labels: `0` non-hate, `1` hate).
  It is **gated**: request access on the dataset page and log in with a Hugging Face token. The data is not redistributed in this repo.
  Check the license and terms of the dataset and its source corpora before use.
- **Size:** 449,078 raw rows, 447,624 after removing 1,454 exact duplicate texts (49 texts appeared with conflicting labels, a sign of annotation noise).
- **Imbalance:** 16,507 hate vs 431,117 non-hate (**3.69% hate**). One source corpus (`jhsc`) contributes ~89% of the rows and has only 1.7% hate,
  while the other sources listed on the dataset page range from ~8% to ~49% hate, so label definitions and base rates differ by source.
- **Splits** (stratified 80/10/10, duplicates removed *before* splitting, leakage checked by an assertion):

| Split | Rows | Hate rate | Notes |
|---|---|---|---|
| Train (full) | 358,099 | 3.69% | |
| Train (balanced) | 26,412 | 50% | undersampled: all 13,206 hate + 13,206 random non-hate. **Used for training** |
| Validation | 44,762 | 3.69% | natural distribution; used for checkpoint and threshold selection |
| Test | 44,763 | 3.69% | natural distribution; used once for the final numbers |

## 3. Approach

```
HF dataset -> clean + dedup -> stratified split -> undersample TRAIN only
          -> AraBERT preprocessing + tokenization (max 128 tokens; median length 20)
          -> fine-tune AraBERT (2 epochs) -> pick best epoch on validation macro-F1
          -> tune decision threshold on VALIDATION, freeze it -> evaluate once on TEST
           -> Gradio demo
```

| Component | Choice |
|---|---|
| Baseline | TF-IDF (word 1-2 grams + char 2-5 grams) + Logistic Regression, same balanced train set |
| Representation | `aubmindlab/bert-base-arabertv02` encoder (135M parameters), official `ArabertPreprocessor` |
| Decision component | Classification head on top of the encoder |
| Fine-tuning | **Full fine-tuning** (not LoRA): lr 2e-5, batch 32, 2 epochs, weight decay 0.01, warmup 6%, fp16/bf16, early stopping (patience 2), seed 42. About 9 minutes on a free Colab T4 |
| Threshold | Chosen to maximize macro-F1 on **validation only** (never on test), then frozen. Applied identically to both models |

**Why undersample and tune a threshold?** With 3.7% positives, a model trained on the raw data mostly predicts "non-hate".
Balancing the training set makes the model sensitive to hate speech, but shifts its probabilities upward: at the default 0.5 threshold,
the model has high recall and very low precision. Tuning the threshold on validation corrects this. The trade-off is that undersampling discards most non-hate examples.

## 4. Results

Test set: 44,763 texts (1,651 hate). The same split, features, and threshold procedure are used for both models.

| Model | Threshold | Accuracy | Macro-F1 | Hate precision | Hate recall | Hate F1 | ROC-AUC | Avg. precision (hate) |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.50 | 0.8708 | 0.6284 | 0.2031 | 0.8565 | 0.3284 | 0.9288 | 0.3601 |
| TF-IDF + LR | 0.804 (val-tuned) | 0.9481 | 0.6938 | 0.3552 | 0.4979 | 0.4146 | 0.9288 | 0.3601 |
| AraBERT | 0.50 | 0.8410 | 0.6024 | 0.1760 | 0.8995 | 0.2944 | 0.9392 | 0.4774 |
| **AraBERT** | **0.947 (val-tuned)** | **0.9637** | **0.7380** | **0.5083** | **0.4821** | **0.4949** | **0.9392** | **0.4774** |


## 5. Demo

A Gradio app takes Arabic text and shows the predicted label, the probabilities, and the decision threshold.

```bash
python -m src.app          # needs the trained model in artifacts/ (see "How to run")
```

<!-- TODO: add a screenshot, e.g. docs/demo.png, and reference it here -->

Sample inputs and outputs are in [`examples/`](examples/sample_outputs.md).

## 6. How to run

### Option A: Google Colab (recommended)

1. Open `notebooks/arabic_hate_speech_training.ipynb` in Colab and select **Runtime -> T4 GPU**.
2. Request access to the gated dataset on Hugging Face, then add your token as a Colab secret named `HF_TOKEN` (or paste it when prompted).
3. Run all cells. The notebook writes `src/` itself, trains (~9 min), evaluates, compares with the baseline, and launches the demo.
   Outputs are saved to Google Drive under `MyDrive/arabic_hate`.

### Option B: command line (GPU strongly recommended)

```bash
git clone https://github.com/<your-username>/<your-repo>.git && cd <your-repo>
pip install -r requirements.txt
huggingface-cli login            # or: hf auth login  (token with access to the gated dataset)

python -m src.train              # prepare data, train, evaluate -> artifacts/ and results/
python -m src.baseline           # TF-IDF baseline + comparison table -> results/baseline_comparison.csv
python -m src.predict "شكرا لك"   # one-off prediction with the saved model
python -m src.app                # Gradio demo
```


Set `PROJECT_OUT=/some/folder` before running to change where `artifacts/` and `results/` are written.

Quick logic tests (no GPU, no dataset access): `python -m pytest tests`




