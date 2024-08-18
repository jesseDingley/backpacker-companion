# Backpacker Companion RAG Application

RAG Application for [thebrokebackpacker.com](https://thebrokebackpacker.com)

## Run it locally

```sh
pip install -e .
streamlit run app.py
```

## Improvements

- additional llm step to check no ooc (if no docs AND LLM extra step says OOC)
- remove 'mate,'
- limit chat history to N turns