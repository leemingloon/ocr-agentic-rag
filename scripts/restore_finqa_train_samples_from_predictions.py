#!/usr/bin/env python3
"""
Restore data/proof/rag/finqa/train/finqa_train_samples.json from
data/proof/rag/finqa/train/finqa_train_predictions.txt.

Reads the .txt only; does not modify it. Output format matches the structure
expected by eval_runner (sample_id, split, ground_truth, input_text, prediction,
prediction_error, metrics, scorer).
"""
from __future__ import annotations

import json
from pathlib import Path


def parse_predictions_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    # Block separator is newline + 72 equals signs + newline
    sep = "\n" + "=" * 72 + "\n"
    blocks = text.split(sep)
    samples = []
    for block in blocks:
        block = block.strip()
        if not block or "sample_id:" not in block:
            continue
        # Split into: header (kv lines), first "--------", prediction section, second "--------", footer (metrics, scorer)
        dash_sep = "-" * 72
        dash_idx = block.find("\n" + dash_sep + "\n")
        if dash_idx == -1:
            continue
        header_text = block[:dash_idx]
        rest = block[dash_idx + len("\n" + dash_sep + "\n"):]
        # Parse header key: value lines
        kv = {}
        for line in header_text.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            kv[key] = value
        sample_id = kv.get("sample_id", "")
        split = kv.get("split", "train")
        # input_text and ground_truth are JSON
        try:
            input_text = json.loads(kv["input_text"]) if kv.get("input_text") else {}
        except json.JSONDecodeError:
            input_text = {"query": kv.get("input_text", "")}
        try:
            ground_truth = json.loads(kv["ground_truth"]) if kv.get("ground_truth") else {}
        except json.JSONDecodeError:
            ground_truth = {}
        # Rest: "prediction:\n" + body + "\n--------\n" + "metrics: ...\nscorer_label: ...\nscorer_note: ..."
        if rest.startswith("prediction:\n"):
            rest = rest[len("prediction:\n"):]
        second_dash = rest.find("\n" + dash_sep + "\n")
        if second_dash == -1:
            continue
        prediction_body = rest[:second_dash].strip()
        footer = rest[second_dash + len("\n" + dash_sep + "\n"):].strip()
        # Parse footer: metrics (JSON), scorer_label, scorer_note
        metrics = {}
        scorer_label = "FAIL"
        scorer_note = ""
        for line in footer.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("metrics:"):
                try:
                    metrics = json.loads(line_stripped.split(":", 1)[1].strip())
                except (json.JSONDecodeError, IndexError):
                    pass
            elif line_stripped.startswith("scorer_label:"):
                scorer_label = line_stripped.split(":", 1)[1].strip() or "FAIL"
            elif line_stripped.startswith("scorer_note:"):
                scorer_note = line_stripped.split(":", 1)[1].strip() if ":" in line_stripped else ""
        row = {
            "sample_id": sample_id,
            "split": split,
            "ground_truth": ground_truth,
            "input_text": input_text,
            "prediction": prediction_body,
            "prediction_error": None,
            "metrics": metrics,
            "scorer": {"label": scorer_label, "note": scorer_note},
        }
        samples.append(row)
    return samples


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    txt_path = base / "data" / "proof" / "rag" / "finqa" / "train" / "finqa_train_predictions.txt"
    out_path = base / "data" / "proof" / "rag" / "finqa" / "train" / "finqa_train_samples.json"
    if not txt_path.exists():
        raise SystemExit(f"Predictions file not found: {txt_path}")
    samples = parse_predictions_txt(txt_path)
    print(f"Parsed {len(samples)} samples from {txt_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
