# class-it-up Agent Guide

This is a project focused on fine-tuning a model for anti-stalking purposes. The project fine-tunes a BERT-based model to classify comments as either "stalking" or "non-stalking". After training, the model is exported to ONNX format for efficient inference. The project also includes a FastAPI application to serve the model for real-time predictions.

## Tech Stack

- **uv**: A tool for managing Python dependencies and virtual environments.
- **FastAPI**: A modern, fast web framework for building APIs with Python.
- **ONNX**: Open Neural Network Exchange format for representing machine learning models.
- **ONNX Runtime**: A high-performance inference engine for ONNX models.
- **Transformers**: A library for natural language processing tasks, providing pre-trained models and tools for fine-tuning.
- **Torch**: A deep learning framework that provides flexibility and speed for building and training models.
- **bert-base-chinese**: A pre-trained BERT model specifically designed for Chinese language processing tasks.

## Structure & Responsibilities

- Training
  - `test.ipynb`: A Jupyter notebook for exporting data, training, testing and exporting the model.
  - `waline.sqlite`: A SQLite database file containing the original data.
  - `dataset.ndjson`: A JSON file containing the processed dataset for training.
  - `fine_tuned_bert`: A directory containing the exported BERT model.
- Model Quantization
  - `src/onnx_convert.py`: A script for converting and quantizing the ONNX model to reduce its size and improve inference speed.
- Model Serving
  - `src/main.py`: A FastAPI application that loads the ONNX model and provides an endpoint for making predictions.

## Commands

- Training:
  - Currently, the training process is conducted within the `test.ipynb` notebook. User will handle it manually by running the cells in the notebook.
- Model Quantization:
  - To quantize the ONNX model, run the following command:
    ```bash
    uv run src/onnx_convert.py
    ```
- Model Serving:
  - To start the FastAPI server, run the following command:
    ```bash
    uv run uvicorn main:app --host 0.0.0.0 --port 8000
    ```
