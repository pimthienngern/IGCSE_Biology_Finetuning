# Stride101 IGCSE Biology Fine-Tuned AI Model

This project showcases a fine-tuned language model designed to answer Edexcel IGCSE Biology questions, focused on Unit 2: Food Production and Ecology. This includes fish farming, ecosystem, eutrophication, greenhouse gases and more.

## Project Overview

I fine-tuned a language model using a custom dataset of structured IGCSE biology questions and answer stored in form of json files. The goal is to assist students in revising biology more interactively through using a chatbot to make learning more approachable.

- **Domain**: Edexcel IGCSE Biology (Unit 2 Food Production)
- **Model**: Fine-tuned from a pretrained transformer model (e.g., DistilBERT, Zephyr)
- **Tech**: Python, Hugging Face Transformers, Gradio
- **Impact**: Aligned with my mission to make education more accessible, especially for underserved communities.

## Files Included

| File             | Description                                      |
|------------------|--------------------------------------------------|
| `train.py`       | Script used to fine-tune the base model          |
| `test.py`        | Used to test the model’s predictions             |
| `data.json`      | Cleaned training data in QA format               |
| `requirements.txt` | Dependencies to run training and testing scripts |

> Note: Model files (e.g., `model.safetensors`) are not included due to GitHub file size limitations.

## Example Question

```bash
Question: Give an example of biological control.
Model Answer: Introducing ladybirds to eat aphids on crops is an example of biological control.
