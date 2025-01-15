# Branch for local dev

This branch provides a simple local working example of the app.

This branch is purely for running small experiments.

The vectorstore used in this branch is produced from roughly 70 blogs and is stored in Git LFS.


## Run app locally

0. 

Create a Python 3.11 virtual environment.

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
- retrieve only relevant chunks / fewer chunks
- Use better LLM (instruct 3)
- Investigate whether sys prompt needs to be passed each time
- Expand Knowledge Base w/ SiteMap
