# Backpacker Companion RAG Application

RAG Application for [thebrokebackpacker.com](https://thebrokebackpacker.com)

## Run it locally

1. 

```sh
cp .dist.env .env
```

and fill-in with your API keys.

2. 

```sh
git lfs pull
pip install -e .
streamlit run app.py
```

## Improvements

- limit chat history to N turns
- prevent repetition at the end of a LLM turn.
- prevent the refusal message from repeating turn after turn.
- return multiple sources if < seuil 