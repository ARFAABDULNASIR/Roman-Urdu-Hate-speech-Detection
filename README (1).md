# Roman Urdu Hate Speech Detection

---

## Overview

A progressive deep learning pipeline for detecting hate speech in Roman Urdu — the Latin-script form of Urdu widely used on social media by 170M+ speakers. The project systematically advances from TF-IDF baselines through custom neural architectures to a multilingual transformer fusion model, ultimately achieving **0.9242 weighted F1** — a **+17.4 point improvement** over the EMNLP 2020 reference benchmark.

---

## The Problem

Roman Urdu hate speech is notoriously difficult to detect for two core reasons:

- **Spelling chaos** — the same word can appear in 5+ spellings simultaneously (e.g. `hai / ha / hy / hey / hae`). Standard NLP treats every variant as a completely different token.
- **Implicit hate** — 63.5% of hate tweets contain zero known hate words. Sentences like *"india k log pagal ha"* require sentence-level semantic understanding, not keyword matching.

English-based platform moderation tools fail entirely on this language, allowing hate speech to go undetected at scale.

---

## Datasets

Four publicly available Roman Urdu corpora were combined:

| Dataset | Size | Labels | Notes |
|---|---|---|---|
| RUHSOLD | 8,008 tweets | Binary + fine-grained | Gold-standard benchmark — EMNLP 2020 |
| HS-RU-20 | 5,000 tweets | Binary (H/N) | 71.5% hate — class imbalance study |
| RU-HSD-30K | 29,999 tweets | Binary (H/N) | Domain expansion |
| RUT | ~72,000 tweets | Binary (Toxic) | Largest labelled corpus; boosts fastText coverage |

After merging, deduplication, and conflict removal:

- **Training:** 89,985 samples
- **Validation:** 14,833 samples
- **Test:** 14,833 samples
- Label distribution: 29.4% hate / 70.6% not-hate

---

## Preprocessing Pipeline

1. **Deduplication** — 17,477 exact duplicates removed to prevent test leakage
2. **Conflict removal** — 543 contradicted rows (same text, different label) dropped
3. **Emoji → text** — `🖕 → _emoji_middle_finger_` (preserves hate signal)
4. **Text normalisation** — lowercase, repeat-collapse (`yaaaar → yaar`), URL/mention removal
5. **Vocabulary** — `min_freq=3`, 89,087 → 34,043 tokens, 94.4% coverage retained
6. **fastText embeddings** — 300-dim vectors trained on 4.7M tweets; 89.1% hit rate
7. **Spelling augmentation** — 30% of training samples augmented with Roman Urdu orthographic variants (`mujhe→mujhay`, `nahi→ni`); adds 20,766 training samples
8. **Zero-leakage split** — augmentation applied only after train/val/test split; verified programmatically

---

## Model Development Journey

| Model | Weighted F1 |
|---|---|
| TF-IDF + LR / SVM (baselines) | 0.896 |
| TextCNN + fastText | 0.903 |
| BiGRU + Attention + fastText | 0.910 |
| DualEncoder (word + char CNN) | 0.906 |
| CNN-BiGRU Cross-Attention | 0.908 |
| **HybridFusion (BiGRU + MuRIL)** | **0.9242** |

All CNN and GRU models share a ~0.91 ceiling. MuRIL integration broke this barrier through pre-trained South Asian multilingual semantics.

---

## Best Architecture: HybridFusion

The final model fuses a BiGRU sequence encoder with [MuRIL](https://huggingface.co/google/muril-base-cased) (Multilingual Representations for Indian Languages), a transformer pre-trained on 11 South Asian languages including their romanised forms.

```
Input Tweet
   ↙              ↘
Word tokens     Subword tokens
fastText(300)   MuRIL tokenizer
    ↓                ↓
BiGRU 2-layer   MuRIL encoder
(h=128, bidir)  bottom 6 frozen
Attn pooling    [CLS] vector
h_seq (256)     h_trans (768)
   ↘              ↙
   Gated Fusion
   g = σ(W·[h_seq; h_trans])
   out = g⊙h_seq + (1−g)⊙h_trans
       → Linear(1024→2)
```

**Why gated fusion?** The gate is computed per-sample — clean text lets MuRIL dominate; noisy spelling shifts weight toward the BiGRU branch.

**Two-phase training:** Phase 1 trains the BiGRU alone (transformer frozen). Phase 2 jointly fine-tunes with differential learning rates (`trans_lr=2e-5 < seq_lr=1e-3`), preventing catastrophic forgetting of MuRIL's South Asian linguistic features.

---

## Training Techniques

**Focal Loss (γ=2)** — Down-weights easy not-hate examples, focusing gradient on hard misclassified hate tweets. Directly addresses the persistent 8–9pt F1(hate) gap caused by the 70.6% not-hate training distribution.

**Differential Learning Rate** — Embedding layer uses `lr=1e-5` while all other parameters use `lr=5e-4`. Embeddings constitute 80–94% of total parameters; a uniform LR overwrites fastText representations within 2 epochs.

**R-Drop (α=0.5)** — Each batch passes through the model twice with different dropout masks. A symmetric KL divergence between the two output distributions is added to the task loss, forcing consistent predictions under uncertainty.

**Layer-wise LR Decay (MuRIL)** — The classifier head gets the full learning rate; each lower transformer layer gets `lr × 0.9^depth`. Bottom layers encode universal South Asian morphology and should change minimally.

**Cosine LR + Warmup** — Linear warmup for 6% of steps, then cosine decay to near-zero. Substantially better than ReduceLROnPlateau in transformer fine-tuning.

---

## Results

| Metric | HybridFusion (Best) | EMNLP 2020 Reference |
|---|---|---|
| Weighted F1 | **0.9242** | 0.75 |
| F1 (hate class) | **0.8729** | — |
| Accuracy | reported in results CSV | — |

The best model improves weighted F1 by **+17.4 percentage points** over the EMNLP 2020 baseline (Rizwan et al., 2020 — the paper that introduced the RUHSOLD dataset).

---

## Key Findings

- Spelling variation and implicit hate **cannot be solved by local-pattern models alone**. TextCNN, BiGRU, and all hybrid CNN/RNN architectures plateau around 0.91 weighted F1 regardless of architecture complexity or training tricks.
- Pre-trained multilingual semantics (MuRIL) was the single breakthrough that crossed the 0.91 ceiling, because it understands sentence meaning without relying on hate keyword presence.
- Multi-task learning using RUHSOLD's fine-grained labels underperformed: only 9.3% of samples carry fine-grained annotations, making the auxiliary loss mostly noise for 90.7% of training.
- The RUT corpus (61% of training data, only 21% hate content) was the primary source of class imbalance and required Focal Loss to mitigate.

---

## Project Structure

```
dl_project/
├── data_preprocessing.ipynb            # Full preprocessing pipeline
├── 01_model_training.ipynb             # Baselines + TextCNN + BiGRU + DualEncoder + CNN-BiGRU
├── 02-improved-training.ipynb          # Focal Loss + Differential LR + R-Drop + XLM-RoBERTa
├── 02_model_training_muril_tweetxlm.ipynb  # MuRIL and TweetXLM experiments
├── 04_model_training_hybrid_fusion.ipynb   # HybridFusion (BiGRU + MuRIL) — best model
└── roman_urdu_presentation.pptx        # Project presentation slides
```

---

## Dependencies

- Python 3.9+
- PyTorch
- Transformers (HuggingFace)
- `google/muril-base-cased`
- scikit-learn
- pandas, numpy, matplotlib, seaborn
- emoji
- fastText embeddings (domain-specific, trained on 4.7M tweets — linked in preprocessing notebook)

---


## Authors

* Muhammad Hunain Ashraf
* Arfa Abdul Nasir 

