# Arabic Sentiment Analysis

### Classical ML vs Transformers vs LLMs vs Hybrid Models

<p align="center">

**A comparative study of Arabic sentiment analysis, sarcasm, and model efficiency**

</p>

> 📄 **Full Project Report:** The complete project report, including the methodology, preprocessing pipeline, model configurations, experimental results, sarcasm analysis, computational trade-offs, limitations, and web application is NLP-FINAL-REPORT.pdf

---

## 📌 Overview

This project presents a comprehensive study of **Arabic Sentiment Analysis** using multiple machine learning and deep learning paradigms.

The study evaluates **seven models across four approaches** on the **LSAHR (Large-Scale Arabic Hotel Reviews)** dataset:

- Classical Machine Learning
- Arabic Transformer Models
- Zero-Shot Arabic Language Models
- Hybrid Ensemble Learning

Rather than focusing only on prediction accuracy, the project investigates the relationship between **performance, computational cost, inference efficiency, and linguistic challenges** in Arabic sentiment analysis.

A particular focus is placed on **Arabic sarcasm and irony**, which remain challenging even for transformer-based models.

The project also includes an interactive **Flask web application** integrating the different models, Arabic OCR, confidence estimation, and real-time model comparison.

---

## 🎯 Research Questions

The study investigates three main questions:

1. **How much do Arabic-specialized transformer models improve sentiment classification compared with classical machine learning approaches?**

2. **How effectively can different model families handle sarcastic and ironic Arabic expressions?**

3. **Is the additional computational cost of transformer models justified by their performance improvement?**

---

# 📊 Dataset

The experiments use the **LSAHR (Large-Scale Arabic Hotel Reviews)** dataset.

The dataset contains:

- **122,679 Arabic hotel reviews**
- Numerical ratings from **1 to 10**
- Hotel and review metadata
- Arabic-language reviews from hotel guests

For this study, the original rating scale was converted into a binary sentiment classification task:

```text
Rating 1–8   → Negative
Rating 9–10  → Positive
```

This resulted in:

- **69,337 Negative reviews**
- **52,229 Positive reviews**

> ⚠️ The LSAHR dataset is **not included** in this repository.

---

# 🧹 Data Preprocessing

Two different preprocessing strategies were implemented depending on the model family.

### Classical Machine Learning Pipeline

The classical models use a more aggressive preprocessing pipeline:

- Arabic language filtering
- Unicode normalization
- Alef and Yeh normalization
- Diacritic removal
- Tatweel removal
- Non-Arabic character removal
- URL and email removal
- Arabic stop-word removal
- TF-IDF feature extraction

### Transformer Pipeline

A lighter preprocessing strategy was used for transformer models.

More of the original text is preserved so that contextual models can learn linguistic information directly from the input sequence.

This distinction allows the study to compare traditional feature-engineering approaches with contextual representation learning.

---

# 🤖 Models

Seven models were evaluated across four different paradigms.

## 1. Classical Machine Learning

### Naive Bayes

Multinomial Naive Bayes using word-level TF-IDF features.

### Logistic Regression

Logistic Regression with:

- L2 regularization
- Balanced class weights
- Word-level TF-IDF features

### SVM

Linear Support Vector Machine using character-level TF-IDF n-grams.

Character n-grams were selected to capture:

- Arabic morphological patterns
- Sub-word information
- Spelling variations
- Dialectal variations

---

## 2. Transformer Models

### AraBERT

An Arabic-specific BERT architecture fine-tuned for binary sentiment classification.

### MARBERT

An Arabic transformer model pretrained on a large collection of Arabic Twitter data and evaluated for sentiment classification.

Both transformer models were fine-tuned using:

- PyTorch
- Hugging Face Transformers
- AdamW
- FP16 training
- Learning-rate scheduling
- Warmup

---

## 3. Zero-Shot Arabic Model

### CAMeL-Lab

A pretrained Arabic sentiment model evaluated on LSAHR **without additional fine-tuning**.

This provides a domain-transfer / zero-shot baseline.

---

## 4. Hybrid Ensemble

### Weighted Voting Ensemble

A hybrid model combining:

- Naive Bayes
- Logistic Regression
- SVM

The voting weights are based on the validation performance of the individual models.

No additional model training is required for the ensemble.

---

# 📈 Experimental Results

All models were evaluated on a held-out test set.

| Model | Accuracy | Precision | Recall | F1 |
|:--|--:|--:|--:|--:|
| Naive Bayes | 70.1% | 69.6% | 68.6% | 68.8% |
| Logistic Regression | 70.3% | 70.1% | 70.5% | 70.1% |
| SVM (Character N-grams) | 70.6% | 70.4% | 70.8% | 70.4% |
| Voting Hybrid | 71.3% | 71.0% | 71.0% | 71.0% |
| **AraBERT** | **73.0%** | **72.7%** | **72.7%** | **72.7%** |
| **MARBERT** | **73.0%** | **72.7%** | **73.1%** | **72.7%** |
| CAMeL-Lab | 62.9% | 61.9% | 60.7% | 60.6% |

---

# 🔎 Key Findings

### Transformer Performance

AraBERT and MARBERT achieved **73.0% accuracy**, compared with **70.6%** for the strongest classical model.

The observed improvement is relatively modest compared with the substantially higher computational requirements of transformer models.

### AraBERT vs MARBERT

Both models achieved the same overall accuracy of **73.0%**.

This suggests that the characteristics of the LSAHR hotel-review dataset may reduce the expected advantage of MARBERT's exposure to informal and dialectal Arabic.

### Hybrid Ensemble

The voting ensemble achieved **71.3% accuracy**, exceeding each individual classical model without requiring additional training.

### Zero-Shot Performance

CAMeL-Lab achieved **62.9% accuracy** without fine-tuning on LSAHR.

This highlights the difficulty of transferring a pretrained Arabic sentiment model to a specific domain such as hotel reviews.

---

# ⚠️ The ~73% Labeling Ceiling

One of the main observations of the study is the convergence of the evaluated models around approximately **73% accuracy**.

The study investigates a potential explanation related to the conversion of the original 10-point rating scale into binary labels.

For example:

```text
Original rating: 8/10
Text: strongly positive
Assigned label: Negative
```

This creates ambiguity between the linguistic content of a review and its assigned ground-truth label.

A review rated 7 or 8 may contain clearly positive language while still being assigned to the Negative class under the chosen labeling rule.

This suggests that part of the observed performance limitation may originate from the **label construction process rather than model capacity alone**.

Potential alternatives include:

- Three-class sentiment classification
- Ordinal sentiment modeling
- Regression over the original rating scale

---

# 🧠 Arabic Sarcasm Analysis

A dedicated analysis was performed to investigate how the different model families respond to sarcastic Arabic expressions.

The evaluation used a small targeted set of sarcastic hotel-review phrases where the literal wording differs from the intended sentiment.

### Main observations

- Classical models frequently rely on surface-level lexical features.
- Transformer models partially handle structurally obvious sarcasm.
- Transformer predictions on sarcastic examples often have relatively low confidence.
- The zero-shot model struggled consistently with the targeted sarcastic examples.
- None of the tested approaches can be considered a reliable standalone Arabic sarcasm detector based on this evaluation.

The analysis suggests that robust sarcasm understanding requires capabilities beyond standard sentiment classification, including contextual, pragmatic, cultural, and world-knowledge reasoning.

---

# ⚙️ Performance vs Computational Cost

The project also evaluates the practical relationship between predictive performance and computational requirements.

| Model | Approx. Size | Training Time | Hardware |
|:--|--:|--:|:--|
| Naive Bayes | <1 MB | 0.02 s | CPU |
| Logistic Regression | <1 MB | 0.71 s | CPU |
| SVM | <1 MB | 18.6 s | CPU |
| Voting Hybrid | <1 MB | No additional training | CPU |
| AraBERT | 540 MB | 28 min | Tesla T4 16 GB |
| MARBERT | 650 MB | 45 min | Tesla T4 16 GB |
| CAMeL-Lab | 440 MB | Zero-shot | — |

The results demonstrate a clear **performance-resource trade-off**:

- Classical models require very little computational resources.
- The hybrid ensemble provides an intermediate performance level without additional training.
- Transformer models achieve higher accuracy but require substantially more computational resources.

This analysis is an important part of the project because model selection depends not only on predictive performance but also on the available deployment resources.

---

# 🌐 Web Application

A web application was developed using **Flask** to provide an interactive environment for testing and comparing the models.

### Main Features

- Arabic sentiment analysis
- Multi-model inference
- Model selection
- Simultaneous model comparison
- Confidence scores
- Inference latency
- Arabic OCR using EasyOCR
- Positive examples
- Negative examples
- Mixed-sentiment examples
- Sarcastic examples
- RTL Arabic interface
- Gemini API comparison

The application allows users to submit Arabic text and observe how different model families respond to the same input.

---

# 🧪 Application Testing

The web application was tested using four main categories.

### Clear Positive

Arabic reviews containing explicit positive language.

### Clear Negative

Arabic reviews containing explicit negative language.

### Mixed Sentiment

Reviews containing both positive and negative clauses.

### Sarcasm

Reviews where the literal wording differs from the intended sentiment.

These tests complement the quantitative evaluation performed on the held-out LSAHR test set.

---

# 🛠️ Technologies

### Programming

- Python
- Jupyter Notebook

### Machine Learning

- Scikit-learn
- NumPy
- Pandas

### Deep Learning / NLP

- PyTorch
- Hugging Face Transformers
- AraBERT
- MARBERT

### NLP Processing

- NLTK
- TF-IDF
- Character n-grams

### Computer Vision / OCR

- EasyOCR
- OpenCV

### Web Development

- Flask
- HTML
- CSS
- JavaScript

### Visualization

- Matplotlib
- Seaborn

---

# ⚠️ Limitations

Several limitations were identified during the study:

- Binary labels derived from ordinal ratings introduce ambiguity.
- LSAHR is specific to the hotel-review domain.
- Sarcasm evaluation uses a small targeted test set.
- Transformer models require substantially more computational resources.
- Zero-shot performance may vary across domains.
- The study does not establish generalization across all Arabic dialects.
- The sarcasm experiments should not be interpreted as a complete benchmark for Arabic sarcasm detection.

---

# 🚀 Future Work

Potential future directions include:

### Better Sentiment Labels

- Three-class sentiment classification
- Ordinal regression
- Preservation of the original rating scale

### Larger Language Models

- Larger Arabic language models
- QLoRA-based fine-tuning
- Parameter-efficient fine-tuning

### Cross-Dataset Evaluation

Evaluation on additional Arabic sentiment datasets such as:

- LABR
- BRAD

### Sarcasm Detection

- Arabic sarcasm-annotated datasets
- Contrastive learning
- Multi-task learning
- Dedicated sarcasm classification

### Explainability

- SHAP
- Integrated Gradients
- Model decision analysis

### Multimodal NLP

Combining:

```text
Text + Images + Emojis
```

to improve sentiment and sarcasm understanding in social-media-style content.

---

# 📄 Documentation

The complete technical documentation is available in:

**[`report/NLP-FINAL-REPORT.pdf`](./report/NLP-FINAL-REPORT.pdf)**

The report contains the full:

- Research motivation
- Related work
- Dataset analysis
- Preprocessing methodology
- Model configurations
- Training configuration
- Experimental results
- Confusion matrix analysis
- Sarcasm analysis
- Computational cost analysis
- Web application
- Limitations
- Future work

---

# 🎓 Academic Context

**Arabic Sentiment Analysis — NLP Final Project**

**Badji Mokhtar – Annaba University**  
Department of Computer Science

**May 2026**

### Author

**Salah Hacen Nasrallah Zaoui**

AI / Computer Science Engineering Student

---

# 📌 Repository Notes

The repository contains the project implementation, experiments, documentation, and supporting resources.

The original LSAHR dataset and large pretrained model weights are not included in the repository.

For reproduction, the required datasets and pretrained models should be obtained through their respective official sources.

---

## License

This project is provided for **educational and research purposes**.
