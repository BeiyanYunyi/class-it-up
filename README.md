# class-it-up

Personal project to classify stalker's spamming comments from other comment.

Using BERT pre-trained model to classify the comments. Has over 95% accuracy, compared to TF-IDF's 80%.

Also provides a dataset of 440+ comments.

## File structure

- server.py
  - FastAPI server to serve the model.
- dataset.ndjson
  - Dataset.
- src/
  - onnx_convert.py
    - Convert the model to ONNX format.
- test.ipynb
  - Contains the data-loading and training code.

## Development

### Requirements

This project uses [uv](https://docs.astral.sh/uv/) to manage the virtual environment. To install the requirements, run:

```bash
uv sync
```
