# Arabic Sentiment Analysis

### A Comparative Study of Classical ML, Transformers, LLMs, and Hybrid Models

> 📄 **Full Project Report:** The complete report documenting the methodology, experiments, model configurations, evaluation, sarcasm analysis, computational trade-offs, limitations, and web application is available in the [`report/`](./report/) folder.

---

## Overview

This project presents a comprehensive study of **Arabic Sentiment Analysis** using multiple machine learning and deep learning paradigms.

The study evaluates classical machine learning models, Arabic-specific transformer architectures, a zero-shot Arabic language model, and a hybrid voting ensemble on the **LSAHR (Large-Scale Arabic Hotel Reviews)** dataset.

The main objective is not only to compare predictive performance, but also to investigate the relationship between:

- Model accuracy
- Generalization
- Computational cost
- Training time
- Inference latency
- Arabic linguistic characteristics
- Sarcasm and pragmatic language

A web application was also developed to provide an interactive environment for testing and comparing the different models.

---

## Research Questions

The project investigates three main questions:

1. How much do Arabic-specialized transformer models improve sentiment classification compared with classical machine learning approaches?

2. How effectively can different model families handle sarcastic and ironic Arabic expressions?

3. Is the additional computational cost of transformer models justified by their performance improvement?

---

## Dataset

The experiments use the **LSAHR (Large-Scale Arabic Hotel Reviews)** dataset.

The dataset contains **122,679 Arabic hotel reviews** with ratings and associated metadata.

Each review contains textual information together with a numerical rating from 1 to 10.

For this study, the task was formulated as binary sentiment classification:

text : 
Rating 1–8  → Negative
Rating 9–10 → Positive

This produces:

69,337 Negative reviews
52,229 Positive reviews

The LSAHR dataset is not included in this repository.

Preprocessing

Two different preprocessing pipelines were implemented.

Classical Machine Learning

The classical models use an extensive preprocessing pipeline including:

Arabic character filtering
Unicode normalization
Diacritic removal
Tatweel removal
Non-Arabic character removal
Arabic stop-word removal
TF-IDF vectorization
Transformer Models

A lighter preprocessing pipeline was used for transformer models in order to preserve more of the original linguistic information.

This allows the contextual models to learn useful representations directly from the input text.

Models

Seven models were evaluated across four different paradigms.

Classical Machine Learning
1. Naive Bayes

Multinomial Naive Bayes using word-level TF-IDF features.

2. Logistic Regression

Logistic Regression with L2 regularization, balanced class weights, and word-level TF-IDF features.

3. SVM

Linear Support Vector Machine using character-level TF-IDF n-grams.

Character n-grams were used to capture sub-word patterns, Arabic morphological variations, and spelling differences.

Transformer Models
4. AraBERT

An Arabic-specific BERT architecture fine-tuned for binary sentiment classification.

5. MARBERT

An Arabic transformer model with extensive pre-training on Arabic Twitter data, evaluated for its ability to handle informal and dialectal Arabic.

Both models were fine-tuned using PyTorch and Hugging Face Transformers.

Zero-Shot Arabic Model
6. CAMeL-Lab

A pre-trained Arabic sentiment model evaluated without additional fine-tuning on the LSAHR dataset.

This provides a domain-transfer / zero-shot baseline.

Hybrid Model
7. Voting Ensemble

A weighted voting ensemble combining:

Naive Bayes
Logistic Regression
SVM

The ensemble weights are proportional to the validation performance of the individual models.

Experimental Results

The models were evaluated on a held-out test set.

Model	Accuracy	Precision	Recall	F1
Naive Bayes	70.1%	69.6%	68.6%	68.8%
Logistic Regression	70.3%	70.1%	70.5%	70.1%
SVM (Character N-grams)	70.6%	70.4%	70.8%	70.4%
Voting Hybrid	71.3%	71.0%	71.0%	71.0%
AraBERT	73.0%	72.7%	72.7%	72.7%
MARBERT	73.0%	72.7%	73.1%	72.7%
CAMeL-Lab	62.9%	61.9%	60.7%	60.6%
Key Findings
Transformer Performance

AraBERT and MARBERT achieved 73.0% accuracy, compared with 70.6% for the strongest classical model.

The improvement was relatively modest considering the substantially larger computational requirements of transformer models.

AraBERT vs MARBERT

Both models achieved 73.0% accuracy.

The similar performance suggests that the characteristics of the LSAHR hotel-review dataset may reduce the expected advantage of MARBERT's exposure to informal and dialectal Arabic.

Hybrid Ensemble

The voting ensemble achieved 71.3% accuracy, outperforming each individual classical model without requiring additional model training.

Zero-Shot Model

CAMeL-Lab achieved 62.9% accuracy without fine-tuning on LSAHR, illustrating the difficulty of transferring a pre-trained sentiment model to a domain-specific Arabic review dataset.

The 73% Labeling Ceiling

One of the central observations of the study is that the evaluated models converge around approximately 73% accuracy.

The main hypothesis investigated is related to the conversion of the original 10-point rating scale into binary sentiment labels.

For example, a review rated 8/10 may contain strongly positive language while being assigned the Negative class under the chosen labeling rule.

This introduces ambiguity directly into the ground-truth labels.

The project therefore identifies the observed performance ceiling as a potential data and labeling limitation, rather than simply a model capacity problem.

A potential alternative would be to preserve more of the original ordinal information through a three-class or regression-based formulation.

Sarcasm Analysis

Arabic sarcasm was evaluated using a small targeted set of sarcastic hotel-review expressions.

The test includes examples where the surface-level wording appears positive or neutral while the intended meaning is negative.

The experiments demonstrate that sarcasm remains difficult across all tested model families.

Observations
Classical models often rely on surface-level lexical features.
Transformer models can partially handle structurally obvious sarcastic expressions.
Predictions on sarcastic examples frequently have relatively low confidence.
The zero-shot model consistently struggled with the targeted sarcastic examples.

The results suggest that robust Arabic sarcasm detection requires additional contextual, pragmatic, and cultural information beyond standard sentiment classification.

Performance vs Computational Cost

An additional objective of the project was to analyze the relationship between predictive performance and computational resources.

Model	Approx. Size	Training Time	Hardware
Naive Bayes	<1 MB	0.02 s	CPU
Logistic Regression	<1 MB	0.71 s	CPU
SVM	<1 MB	18.6 s	CPU
Hybrid	<1 MB	No additional training	CPU
AraBERT	540 MB	28 min	Tesla T4 16 GB
MARBERT	650 MB	45 min	Tesla T4 16 GB
CAMeL-Lab	440 MB	Zero-shot	—

The comparison highlights the practical trade-off between model capacity, predictive performance, and computational requirements.

Web Application

A web application was developed using Flask to demonstrate the different models in an interactive environment.

Features
Arabic text sentiment analysis
Multi-model inference
Model comparison
Confidence scores
Inference latency measurement
Arabic OCR using EasyOCR
Quick-test examples
Positive, negative, mixed, and sarcastic examples
RTL Arabic interface
Gemini API comparison

The application allows users to submit Arabic text and compare the predictions and confidence scores of multiple models.

Application Testing

The web application was evaluated using several categories of Arabic text.

Clear Positive

Reviews containing explicit positive language.

Clear Negative

Reviews containing explicit negative language.

Mixed Sentiment

Reviews containing both positive and negative clauses.

Sarcasm

Expressions where the literal wording differs from the intended sentiment.

These tests complement the quantitative evaluation performed on the held-out dataset.

Limitations

Several limitations were identified during the project:

Binary labels derived from ordinal ratings introduce ambiguity.
LSAHR is domain-specific to hotel reviews.
Sarcasm evaluation uses a small targeted test set.
Transformer models require substantially more computational resources.
Zero-shot performance may vary across domains.
The study does not establish generalization across all Arabic dialects.
Future Work

Potential directions include:

Three-class sentiment classification
Ordinal sentiment modeling
Larger Arabic language models
QLoRA-based fine-tuning
Cross-dataset evaluation
Arabic sarcasm datasets
Contrastive learning
Multi-task learning
Explainability using SHAP and Integrated Gradients
Multimodal sentiment analysis using text, images, and emojis
