# Name Generator

A small Transformer that invents new names. I built and trained it in the Transformers workshop of the course *Artificial Intelligence and Machine Learning*.

**Try it:** PASTE-YOUR-APP-LINK-HERE

## What it does

The model learned from a list of names by practising one task: predict the next letter. To invent a name it spells one letter at a time, and the temperature slider decides how adventurous each choice is.

## What is in this repository

| File | What it is |
|---|---|
| `app.py` | the web app (Streamlit) and the model code |
| `model.pt` | the trained parameters, the alphabet, and the list of names |
| `requirements.txt` | the packages the app needs |

## The model

- A decoder-only Transformer (the same design as GPT), written in PyTorch.
- 2 blocks, 2 attention heads per block, vectors of length 32, about 19,000 parameters.
- Trained on a laptop in less than a minute.

## Run it on your own computer

```
pip install streamlit torch
streamlit run app.py
```
