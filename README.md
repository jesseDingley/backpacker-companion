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

- additional llm step to check no ooc (if no docs AND LLM extra step says OOC)
- remove 'mate,'
- limit chat history to N turns