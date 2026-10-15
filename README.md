# Arabic Hate Speech Classifier (AraBERT)

Binary classification of Arabic text as **Hate Speech** or **Non-Hate Speech**. A fine-tuned AraBERT model is compared with a
TF-IDF + Logistic Regression baseline on the same held-out test set, and served through a small Gradio demo.

> **Content warning:** the dataset and the error analysis contain offensive and hateful Arabic text.

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
          -> error analysis -> Gradio demo
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

Raw files: [`results/baseline_comparison.csv`](results/baseline_comparison.csv), [`results/classification_report.txt`](results/classification_report.txt).

How to read this:

- With a tuned threshold, AraBERT beats the baseline on macro-F1 (0.738 vs 0.694) and hate F1 (0.495 vs 0.415), mainly through higher precision (0.51 vs 0.36) at similar recall.
- The threshold-free metrics agree in direction: average precision for the hate class is 0.477 vs 0.360, and ROC-AUC is 0.939 vs 0.929.
  The ROC-AUC gap is small, and this is a single run with one seed and no confidence intervals, so treat the size of the gap with caution.
- At the default 0.5 threshold, AraBERT is *not* better on macro-F1 (0.602 vs 0.628). The improvement only appears once the decision threshold is calibrated for the real 3.7% base rate.
- The task is still hard: at the chosen threshold, about half of the hate speech in the test set is missed (recall 0.48) and about half of the flagged texts are not labeled hate (precision 0.51).
- Validation: the best checkpoint was epoch 2 (validation macro-F1 0.6046 at threshold 0.5; ROC-AUC 0.9395).

![Confusion matrix](results/confusion_matrix.png)

## 5. Error analysis

At the tuned threshold the model makes 3,285 errors on the test set: 1,700 false positives and 1,585 false negatives.
`error_analysis.csv` holds a **random sample of 15 false positives and 15 false negatives** (not cherry-picked), with simple automatic hints
(very short text, negation marker, character elongation, Latin letters, mention/URL, high-confidence error).

<!-- TODO before submitting: open results/error_analysis.csv, read at least 10 failures, fill the human_review_category column,
     then replace this table with your real counts and one example per category (avoid pasting slurs; shorten or mask them). -->

| Category (assigned by manual review) | Type | Count (of 30 sampled) | What happens |
|---|---|---|---|
| TODO e.g. insult without a protected target | FP | TODO | TODO |
| TODO e.g. dialect / spelling variation | FN | TODO | TODO |
| TODO e.g. sarcasm or indirect meaning | FN | TODO | TODO |
| TODO e.g. annotation inconsistency | FP/FN | TODO | TODO |

## 6. Demo

A Gradio app takes Arabic text and shows the predicted label, the probabilities, and the decision threshold.

```bash
python -m src.app          # needs the trained model in artifacts/ (see "How to run")
```

<!-- TODO: add a screenshot, e.g. docs/demo.png, and reference it here -->

Sample inputs and outputs are in [`examples/`](examples/sample_outputs.md).

## 7. How to run

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

Model weights (~540 MB) are not stored in git. <!-- TODO: add a Hugging Face Hub or Drive link to the trained model folder -->

Quick logic tests (no GPU, no dataset access): `python -m pytest tests`

## 8. Repository structure

```
src/
  config.py            paths, model name, hyper-parameters
  data_preparation.py  load, clean, dedup, split, undersample, tokenize
  train.py             Trainer setup, saving artifacts, end-to-end main()
  evaluate.py          metrics, threshold tuning, error table, model card
  baseline.py          TF-IDF + Logistic Regression and the comparison table
  predict.py           HateSpeechClassifier for inference
  app.py               Gradio demo
notebooks/             Colab notebook that runs the whole pipeline
results/               metrics, comparison table, classification report, confusion matrix, error sample
examples/              sample inputs and outputs
tests/                 fast logic tests on synthetic data
```

## 9. Limitations and responsible use

- **Annotation noise and source bias:** the superset mixes corpora with different guidelines, and one corpus dominates. Results were not broken down per source or per dialect.
- **Context:** sarcasm, quotation, religious or political discussion, and dialectal spelling are hard; short texts carry little context.
- **Calibration:** probabilities are not calibrated because of undersampling; use the tuned threshold, not 0.5.
- **Single run:** one seed, no confidence intervals, no cross-validation.
- **Use:** this is a research and course project. Predictions are not definitive judgments and must not be used for automated enforcement without human review.

## 10. Acknowledgements

Base model: [AraBERT](https://github.com/aub-mind/arabert) (AUB MIND Lab). Dataset: `manueltonneau/arabic-hate-speech-superset` and the original corpora it aggregates.
