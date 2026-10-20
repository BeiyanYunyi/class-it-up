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

## CLI

Run these commands from the repository root after `uv sync` (the default dev
dependencies include Polars, Torch and scikit-learn). Paths are relative to the
current working directory. The four commands replace cells L1, L2, L4 and the
ONNX conversion script respectively:

```bash
# L1: export comments; status == "waiting" maps to label 1, all other statuses to 0
uv run src/cli.py export --database waline.sqlite --output dataset.ndjson

# L2: read NDJSON into a Polars DataFrame, strip HTML tags, and display it
uv run src/cli.py dataframe --input dataset.ndjson
# Optionally persist the cleaned DataFrame as NDJSON
uv run src/cli.py dataframe --input dataset.ndjson --output cleaned.ndjson

# L4: load and clean NDJSON, train BERT, evaluate each epoch, save model + tokenizer
uv run src/cli.py train --input dataset.ndjson --output fine_tuned_bert

# Export ONNX, preprocess it, and dynamically quantize weights to INT8
uv run src/cli.py quantize --model fine_tuned_bert --output-dir .
```

Training defaults match the notebook: `bert-base-chinese`, 2 epochs, batch size
16, learning rate `2e-5`, validation fraction `0.2`, and maximum token length 128.
The device is selected in CUDA → MPS → CPU order. Override with `--device cpu`,
`--epochs`, `--batch-size`, `--learning-rate`, `--test-size`, `--max-length`, or
`--model` (a Hugging Face name or local model directory). `--seed 42` sets the
split and Torch seeds; by default the split is random. Loading a remote model
may download its weights and tokenizer.

Quantization writes `bert_model.onnx`, `bert_model_preprocess.onnx`, and
`bert_model_quantized.onnx` into the output directory. The sequence length is
fixed at `--max-length` (128 by default), while batch size is dynamic; inference
must use the same sequence length. Existing output files/model directories are
overwritten. The old `uv run src/onnx_convert.py` entry point remains available
and accepts the same options as `quantize`.

Use `uv run src/cli.py <command> --help` for all options. The notebook remains
available for exploratory work; the TF-IDF/SVM and prediction cells are not
part of this CLI.

Run the data preparation and CLI checks with:

```bash
uv run python -m unittest discover -s tests
```
