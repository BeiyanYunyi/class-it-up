"""Export a saved BERT classifier and dynamically quantize its ONNX weights."""

from pathlib import Path


def quantize(model_path: Path, output_dir: Path, max_length: int = 128):
    import torch
    import onnx
    from transformers import BertForSequenceClassification
    from onnxruntime.quantization import quantize_dynamic, preprocess, QuantType

    model = BertForSequenceClassification.from_pretrained(model_path).cpu().eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    original = output_dir / "bert_model.onnx"
    preprocessed = output_dir / "bert_model_preprocess.onnx"
    quantized = output_dir / "bert_model_quantized.onnx"
    input_ids = torch.zeros((1, max_length), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(original),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
        opset_version=20,
        dynamo=False,
    )
    preprocess.quant_pre_process(
        input_model=onnx.load(original), output_model_path=str(preprocessed)
    )
    quantize_dynamic(
        model_input=str(preprocessed),
        model_output=str(quantized),
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to {quantized}")


if __name__ == "__main__":
    # Preserve the existing script entry point with the shared CLI options.
    import sys
    from cli import main

    main(["quantize", *sys.argv[1:]])
