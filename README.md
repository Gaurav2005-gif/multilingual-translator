# 🌐 Multilingual Translator

A simple web app to translate text between 100+ languages. Pick the source and target language, type your text, and press Translate.

**Live demo:** https://gaurav-translator.streamlit.app

## Features
- 100+ languages, including Hindi, Tamil, Malayalam, Telugu, French, Spanish and more
- Automatic source-language detection
- One-click language swap
- Multiple translation backends with automatic fallback: Google Translate, then MyMemory, then an offline NLLB-200 model (local use only)

## Tech stack
Python, Streamlit, deep-translator, langdetect, Hugging Face Transformers (NLLB-200), FAISS, sentence-transformers

## Project files
- `web1.py` - Streamlit web app (the deployed version)
- `bot2.py` - command-line translator with offline fallback
- `app.py` - experimental RAG translation-memory demo (FAISS + NLLB-200), runs locally

## Run locally

    pip install streamlit deep-translator langdetect
    streamlit run web1.py

For the offline fallback and the RAG demo, also install: `torch transformers sentence-transformers faiss-cpu sentencepiece`

## Author
Gaurav Vashishth
