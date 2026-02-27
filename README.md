---
title: Wise Cart Assistant
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: false
---

# 🛒 Wise Cart Assistant

An AI-powered shopping assistant that recommends products from an Amazon dataset using RAG (Retrieval-Augmented Generation).

## Setup

### 1. Add your secret
Go to **Settings → Secrets** in your HuggingFace Space and add:

| Secret name | Value |
|-------------|-------|
| GROQ_API_KEY | Your Groq API key |

### 2. Upload your dataset
Upload `amazon_product_data.csv` to the **root** of this Space repository.

### 3. Deploy
Push all files and the Space will build automatically.

## Files

| File | Purpose |
|------|---------|
| app.py | Main application |
| requirements.txt | Python dependencies |
| amazon_product_data.csv | Product dataset (you upload this) |
