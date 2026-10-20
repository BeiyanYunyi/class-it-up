import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_dataframe


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "src/cli.py"), *map(str, args)],
            capture_output=True, text=True,
        )

    def test_export_and_dataframe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "waline.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE wl_Comment (comment TEXT, status TEXT)")
                connection.executemany("INSERT INTO wl_Comment VALUES (?, ?)", [
                    ("<p>你好</p>", "waiting"), ("普通评论", "approved"),
                    ("<b>其他</b>", "spam"),
                ])
            raw = root / "nested/raw.ndjson"
            result = self.run_cli("export", "--database", database, "--output", raw)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            self.assertEqual([row["label"] for row in rows], [1, 0, 0])
            self.assertEqual(rows[0]["comment"], "<p>你好</p>")
            cleaned = root / "cleaned.ndjson"
            result = self.run_cli("dataframe", "--input", raw, "--output", cleaned)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(load_dataframe(cleaned)["comment"].to_list(), ["你好", "普通评论", "其他"])

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "missing.sqlite"
            result = self.run_cli("export", "--database", database)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(database.exists())

    def test_invalid_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.ndjson"
            for label in [2, -1, 0.5, None]:
                source.write_text(json.dumps({"comment": "text", "label": label}) + "\n")
                with self.assertRaises(ValueError):
                    load_dataframe(source)

    def test_training_arguments_are_validated(self):
        for option, value in [("--epochs", "0"), ("--batch-size", "-1"), ("--test-size", "1"), ("--learning-rate", "nan")]:
            result = self.run_cli("train", option, value)
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_all_help_commands(self):
        for command in ["export", "dataframe", "train", "quantize"]:
            result = self.run_cli(command, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
