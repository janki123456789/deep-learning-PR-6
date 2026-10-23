<div align="center">

# 🔍 Opening the Transformer
### Deep Learning Practical Report — PR 6

**Attention • Encoder • Decoder • Inference with Pretrained Models**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

*A practical exploration of how Transformer models process, understand, and generate text.*

</div>

---

## 📌 Project Overview

This project explores the core ideas behind **Transformer architecture** using small, publicly available pretrained models from the Hugging Face Hub. It combines short theory explanations, NumPy implementations, attention visualizations, model inspection, inference experiments, and a mini text-processing application.

The practical focuses on understanding **how Transformers work**, rather than training or fine-tuning a model. The experiments are designed to run on a regular laptop CPU, although downloading models and running inference may take some time.

## 🎯 Objectives

- Understand **Query, Key, and Value (Q, K, V)** and scaled dot-product attention.
- Compare self-attention with Luong attention and explore multi-head attention.
- Understand positional encoding, learned position embeddings, and Layer Normalization.
- Compare bidirectional encoder attention with causal/masked decoder attention.
- Explore cross-attention, autoregressive generation, decoding strategies, and KV-cache.
- Evaluate pretrained sentiment classification and examine practical limitations such as truncation, processing time, and hallucinations.
- Build a small application for sentiment analysis, summarization, and English-to-French translation.

## 🧠 Pretrained Models Used

| Model | Transformer family | Purpose in this practical |
|---|---|---|
| [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | Encoder-only | Sentiment analysis, encoder structure, and attention inspection |
| [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) | Decoder-only | Causal attention, text generation, decoding strategies, and inference |
| [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small) | Encoder–decoder | English-to-French translation, summarization, and cross-attention |

> **Note:** The notebook should reflect the exact model IDs successfully used in your own environment. If you substituted a model, update this table to match.

## 🗂️ Tasks Covered

### 1. Introduction, Pipelines & Tokenizer
- Run a pretrained sentiment-analysis pipeline on sample sentences.
- Review the development of attention and Transformer models.
- Inspect tokens, token IDs, attention masks, padding, and model configuration.

### 2. Self-Attention & Scaled Dot-Product Attention
- Implement attention with NumPy using Q, K, and V.
- Visualize attention weights and verify that each row sums to approximately 1.
- Compare scaled and unscaled scores, implement Luong dot attention, and demonstrate order-blindness.
- Compare a manual attention calculation with a real model's attention output.

### 3. Multi-Head Attention, Position & Layer Normalization
- Inspect real attention heads and visualize attention patterns.
- Compare attention-head entropy.
- Implement sinusoidal positional encoding and LayerNorm from scratch.
- Explore learned positional embeddings and compare LayerNorm with BatchNorm.

### 4. Encoder Architecture & Masked Attention
- Inspect the DistilBERT encoder and its parameter groups.
- Use fill-mask predictions and visualize bidirectional attention.
- Build a causal mask with NumPy and inspect GPT-2's causal attention.
- Compare padding masks with causal masks.

### 5. Cross-Attention, Decoder & Inference
- Explore the Transformer decoder and cross-attention.
- Use T5 for translation and summarization and visualize cross-attention.
- Compare manual greedy decoding with `generate()`.
- Examine greedy decoding, beam search, sampling, temperature, top-k/top-p, and KV-cache timing.

### 6. Evaluation & Mini Application
- Evaluate pretrained DistilBERT on 200 IMDB reviews using accuracy, precision, recall, F1-score, and a confusion matrix.
- Compare input truncation lengths of 64, 128, 256, and 512 tokens.
- Measure inference time at different sequence lengths.
- Check GPT-2 factual continuations and discuss hallucinations.
- Build `text_toolkit(text)` for sentiment analysis, summarization, and French translation.
- Reflect on the strengths and limitations observed during the experiments.


## 🧰 Technologies

- **Python** — implementation and experimentation
- **Hugging Face Transformers** — pretrained Transformer models and pipelines
- **PyTorch** — model inference and tensor operations
- **Datasets** — IMDB dataset loading
- **NumPy** — attention and normalization demonstrations
- **Matplotlib / Seaborn** — plots and heatmaps
- **Pandas / scikit-learn** — result tables and evaluation metrics
- **Jupyter Notebook** — practical workflow and documentation

## 📁 Suggested Repository Structure

```text
your-repository/
├── DL_PR6.ipynb
├── DL_PR6.html
├── requirements.txt
├── README.md
├── plots/
│   ├── attention_scratch.png
│   ├── attention_heads.png
│   ├── positional_encoding.png
│   ├── causal_mask.png
│   ├── cross_attention.png
│   ├── truncation_accuracy.png
│   └── time_vs_length.png
└── video/
    └── (optional: keep your video link in the README)
```

## 🎥 Project Video Demonstration

**Video link:** `ADD_YOUR_GOOGLE_DRIVE_OR_UNLISTED_YOUTUBE_LINK_HERE`


## 💡 Key Learnings

This practical connects the mathematical idea of attention with the behaviour of real pretrained models. It demonstrates how encoder-only, decoder-only, and encoder–decoder architectures serve different purposes, and how masks, positional information, and inference settings affect model behaviour. It also highlights why model outputs and performance measurements need to be checked rather than assumed.

## 👩‍💻 Author

**Janki Dholariya**  
AI/ML & Data Science Student

---