"""Command-line entry point for data preparation, training and quantization."""

import argparse
from pathlib import Path


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def positive_float(value):
    import math

    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def split_fraction(value):
    result = positive_float(value)
    if result >= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="Convert Waline SQLite comments to NDJSON")
    export.add_argument("--database", type=Path, default=Path("waline.sqlite"))
    export.add_argument("--output", type=Path, default=Path("dataset.ndjson"))

    dataframe = commands.add_parser("dataframe", help="Load and display the cleaned Polars DataFrame")
    dataframe.add_argument("--input", type=Path, default=Path("dataset.ndjson"))
    dataframe.add_argument("--output", type=Path, help="Optionally save the cleaned frame as NDJSON")

    train = commands.add_parser("train", help="Fine-tune BERT using NDJSON comments")
    train.add_argument("--input", type=Path, default=Path("dataset.ndjson"))
    train.add_argument("--output", type=Path, default=Path("fine_tuned_bert"))
    train.add_argument("--model", default="bert-base-chinese", help="Hugging Face model name or local directory")
    train.add_argument("--epochs", type=positive_int, default=2)
    train.add_argument("--batch-size", type=positive_int, default=16)
    train.add_argument("--learning-rate", type=positive_float, default=2e-5)
    train.add_argument("--test-size", type=split_fraction, default=0.2)
    train.add_argument("--max-length", type=positive_int, default=128)
    train.add_argument("--seed", type=int, default=None, help="Optional seed for splitting and training")
    train.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    quantize = commands.add_parser("quantize", help="Export saved BERT to ONNX and quantize to INT8")
    quantize.add_argument("--model", type=Path, default=Path("fine_tuned_bert"))
    quantize.add_argument("--output-dir", type=Path, default=Path("."))
    quantize.add_argument("--max-length", type=positive_int, default=128)

    args = parser.parse_args(argv)
    # Load the ML stack only for commands that actually need it.
    try:
        if args.command == "export":
            from data import export_dataset

            frame = export_dataset(args.database, args.output)
            print(f"Exported {frame.height} rows to {args.output}")
        elif args.command == "dataframe":
            from data import load_dataframe

            frame = load_dataframe(args.input)
            print(frame)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                frame.write_ndjson(args.output)
        elif args.command == "train":
            from training import train

            train(args)
        elif args.command == "quantize":
            from onnx_convert import quantize

            quantize(args.model, args.output_dir, args.max_length)
    except (ValueError, OSError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
