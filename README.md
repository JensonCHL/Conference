# Cross-Domain Detection of AI-Generated Songs

Official implementation and experimental results for the paper:

**"Cross Domain Detection of AI Generated Songs: Benchmarking Encoder Models on Bahasa Indonesia"**

**Authors:** Jenson Christopher Halim, Henry Lucky  
**Institution:** Bina Nusantara University, Jakarta, Indonesia

---

## 📋 Overview

This repository contains the complete experimental code for benchmarking three audio encoder models (AST, Wav2Vec 2.0, WavLM) on AI-generated song detection in Bahasa Indonesia, with cross-domain evaluation on speech data.

### Key Contributions

- **First Bahasa Indonesia AI-generated music dataset** (~50 hours)
- **Cross-domain speech dataset** using FineVoice voice cloning
- **Comprehensive benchmarking** of three state-of-the-art audio encoders
- **Cross-domain generalization analysis** from music to speech

---

## 🗂️ Repository Structure

```
Conference/
├── FineTuneMusic/          # Fine-tuning & trainning code notebooks for music dataset
│   ├── trainClassification.ipynb  (AST)
│   ├── wave2vec.ipynb
│   └── wavlm.ipynb
│
├── ZeroShotResult/         # Zero-shot evaluation on music dataset
│   ├── ast-zero-shot/
│   ├── wav2vec2-zero-shot/
│   └── wavlm-zero-shot/
│
├── Fine-TunedSpeech/       # Cross-domain: Fine-tuned models evaluated on speech
│   ├── ast-finetuned-eval/
│   ├── wav2vec2-finetuned-eval/
│   └── wavlm-zero-shot/
│
└── ZershotSPeech/          # Zero-shot evaluation on speech dataset
    ├── ast-zero-shot/
    └── wav2vec2-zero-shot/
    └── wavlm-zero-shot/
```

### File Types in Each Directory

- `*.ipynb` - Jupyter notebooks with training/evaluation code
- `classification_report_test.txt` - Detailed metrics (precision, recall, F1)
- `classification_report_test.csv` - Same metrics in CSV format
- `confusion_matrix_test.csv` - Confusion matrices
- `runs/` - TensorBoard logs (training curves)

---

## 🎯 Results & Output Files

Each experiment directory contains the following output files:

### 📁 Directory Contents

| Directory           | Description                                | What It Contains                                                                               |
| ------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `ZeroShotResult/`   | Zero-shot evaluation on **Music** dataset  | Pre-trained models with randomly initialized classification heads evaluated on music test set  |
| `ZershotSPeech/`    | Zero-shot evaluation on **Speech** dataset | Pre-trained models with randomly initialized classification heads evaluated on speech test set |
| `Fine-TunedSpeech/` | Cross-domain evaluation                    | Models fine-tuned on music, then evaluated on speech (unseen domain)                           |
| `FineTuneMusic/`    | Training notebooks                         | Jupyter notebooks for fine-tuning each model on the music dataset                              |

### 📄 CSV File Contents

**`classification_report_test.csv`** - Contains per-class metrics:

- `precision` - How many predicted positives are actually positive
- `recall` - How many actual positives are correctly predicted
- `f1-score` - Harmonic mean of precision and recall
- `support` - Number of samples per class
- Includes `Real (0)`, `Fake (1)`, `accuracy`, `macro avg`, and `weighted avg`

**`confusion_matrix_test.csv`** - Shows prediction breakdown:

- Rows: True labels (Real, Fake)
- Columns: Predicted labels (Real, Fake)
- Values: Count of samples in each category

**`runs/`** - TensorBoard logs for visualizing training curves

---

## 🔧 Models Used

| Model           | Checkpoint                                | Input Type   | Sampling Rate |
| --------------- | ----------------------------------------- | ------------ | ------------- |
| **AST**         | `MIT/ast-finetuned-audioset-10-10-0.4593` | Spectrogram  | 32 kHz        |
| **Wav2Vec 2.0** | `facebook/wav2vec2-base`                  | Raw waveform | 16 kHz        |
| **WavLM**       | `microsoft/wavlm-base`                    | Raw waveform | 16 kHz        |

---

## ⚙️ Training Configuration

All models were fine-tuned with the following hyperparameters:

```python
Optimizer: AdamW
Learning Rate: 3e-5
Epochs: 3
Batch Size: 4 per device
Weight Decay: 0.01
Audio Duration: 10 seconds (fixed-length chunks)
Dataset Split: 80% train / 10% validation / 10% test
```

---

## 📊 Dataset Information

### Music Dataset

- **Real Songs:** 372 unique titles, 9,671 chunks (~27 hours)
- **AI-Generated:** 54 unique titles, 8,413 chunks (~23 hours)
- **Source:** YouTube (real) and Suno AI (generated)

### Speech Dataset (Cross-Domain)

- **6 unique speakers** using FineVoice voice cloning
- **~1,302 real / ~1,242 fake** samples
- **Speaker-independent split** (no speaker overlap between splits)

> **Note:** Raw audio files are not redistributed due to copyright. The dataset release is limited to metadata and source code to support reproducibility.

---

## 🚀 Usage

### Requirements

```bash
pip install torch torchaudio transformers evaluate scikit-learn pandas numpy
```

### Running Experiments

1. **Fine-tuning on Music:**

   ```bash
   jupyter notebook FineTuneMusic/wavlm.ipynb
   ```

2. **Zero-shot Evaluation on Music:**

   ```bash
   jupyter notebook ZeroShotResult/wavlm-zero-shot/wavlm.ipynb
   ```

3. **Cross-Domain Evaluation (Fine-tuned on Music → Evaluated on Speech):**
   ```bash
   jupyter notebook Fine-TunedSpeech/ast-finetuned-eval/astEvaluate.ipynb
   ```

---

## 📧 Contact

For questions or collaboration inquiries, please contact: j ensonc.haliM@gmail.com

---

## 🙏 Acknowledgments

- **AST:** MIT CSAIL
- **Wav2Vec 2.0:** Meta AI
- **WavLM:** Microsoft Research
- **Suno AI:** For AI-generated music samples
- **FineVoice:** For voice cloning technology
