"""
Dataset adapters for loading and evaluating datasets in a standardized way.
Each adapter handles a specific dataset and implements the logic to load its data, 
parse it into a universal format, and evaluate predictions against ground truth.
The BaseDatasetAdapter defines the common interface and utilities for all datasets, 
while each specific dataset adapter (e.g., SROIEAdapter, FUNSDAdapter) implements 
the dataset-specific directory path, file type loading, and parsing data schema logic.
"""
import ast
import os
import io
import re
import json
import numpy as np
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List
import pyarrow.parquet as pq
from collections import defaultdict
from difflib import SequenceMatcher
from datasets import load_dataset, load_from_disk
from scipy.optimize import linear_sum_assignment


def _parse_options_list(value):  # noqa: ANN001, ANN201
    """Parse options from parquet (string like \"['$63,020', ...]\" or list) into a Python list. Returns None if missing/invalid."""
    if value is None:
        return None
    if isinstance(value, list):
        return value if value else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            out = ast.literal_eval(s)
            return out if isinstance(out, list) and out else None
        except (ValueError, SyntaxError):
            return None
    return None


# ======================================================
# Base Adapter
# ======================================================
class BaseDatasetAdapter:
    """Base class for all dataset adapters."""
    def __init__(
        self,
        category: str,
        dataset_name: str,
        data_source_from_hf_or_manual: str,
        hf_repo_name: str = None,
        hf_repo_variant: str = None
    ):
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.root_dir = os.path.join(os.getcwd(), "data", self.category, self.dataset_name)
        self.dataset_obj = None  # populated for HF datasets
        self.dataset_split = "train"  # default
    
    def download(self):
        if self.data_source_from_hf_or_manual != "hf":
            return

        # If hf_repo_variant is a list, download each variant separately
        if isinstance(self.hf_repo_variant, list):
            self.dataset_obj = {}
            for variant in self.hf_repo_variant:
                print(f"Downloading HF dataset {self.dataset_name}, variant: {variant} ...")
                ds = load_dataset(path=self.hf_repo_name, name=variant)
                
                # Save to disk under subfolder named after variant
                variant_folder = os.path.join(self.root_dir, variant)
                os.makedirs(variant_folder, exist_ok=True)
                ds.save_to_disk(variant_folder)
                print(f"✅ Saved {self.dataset_name}-{variant} to {variant_folder}")
                
                # Store loaded dataset object in dict
                self.dataset_obj[variant] = ds
        else:
            # Original behavior: single variant (string or None)
            dataset_args = {"path": self.hf_repo_name}
            if self.hf_repo_variant:
                dataset_args["name"] = self.hf_repo_variant
            print(f"Downloading HF dataset {self.dataset_name} ...")
            self.dataset_obj = load_dataset(**dataset_args)
            print(f"✅ HF dataset {self.dataset_name} loaded")

    def cleanup(self):
        """Delete downloaded dataset folder, only if data_source_from_hf_or_manual="hf"."""
        if self.data_source_from_hf_or_manual == "hf" and os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)
            print(f"[INFO] Cleaned up dataset {self.dataset_name}")

    def load_split(self, dataset_split="train", recursive=False, max_samples_per_split=None, max_samples_per_category=None):
        """Return list of samples for evaluation. Must be implemented by leaf-level adapters."""
        raise NotImplementedError
    
    def _prepare_samples(
        self,
        dataset_split_obj,
        dataset_name: str,
        input_key: str | dict,
        answer_key: str,
        max_samples_per_split: int | None = None,
    ) -> list[dict]:
        """
        input_key can be:
            - string: maps to single input field
            - dict: {desired_field_name: source_field_name} for multiple fields
        """
        samples = []
        for i, row in enumerate(dataset_split_obj):
            if max_samples_per_split is not None and i >= max_samples_per_split:
                break

            if isinstance(input_key, str):
                input_dict = {input_key: row.get(input_key)}
            elif isinstance(input_key, dict):
                input_dict = {k: row.get(v) for k, v in input_key.items()}
            else:
                input_dict = {}

            # handle lists with fallback to first element
            ans = row.get(answer_key)
            if isinstance(ans, list):
                ans = ans[0] if ans else None

            samples.append({
                "id": f"{dataset_name}_{i}",
                "input": input_dict,
                "answer": ans,
                "metadata": {}
            })
        return samples
    
    # -----------------------------------------
    # Generic helper: load Parquet folder
    # -----------------------------------------
    def _load_arrow_folder(self, folder_path: Path, max_samples_per_split=None, max_samples_per_category=None):
        """Yield parquet rows from a split folder as a generator (streaming)."""
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        max_samples = None
        if max_samples_per_split is not None and max_samples_per_category is not None:
            max_samples = min(max_samples_per_split, max_samples_per_category)
        elif max_samples_per_split is not None:
            max_samples = max_samples_per_split
        elif max_samples_per_category is not None:
            max_samples = max_samples_per_category

        # Delegate to the low-level row streamer. Callers should treat the return
        # value as an iterator/generator and MUST NOT materialize it into a list
        # for large datasets, to avoid exhausting memory on very large parquet
        # shards.
        return self._arrow_row_stream(folder_path, max_samples=max_samples)
    
    @classmethod
    def load_dataset_splits(cls, data_dir=None, json_file="data_structure.json"):
        """Load precomputed dataset splits from JSON"""
        if not data_dir:
            data_dir = os.path.join(os.getcwd(), "data")
        json_path = os.path.join(data_dir, json_file)
        if not os.path.exists(json_path):
            print(f"⚠️ Dataset splits JSON not found at {json_path}")
            return {}
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_available_splits(self):
        """Return available splits for this dataset from JSON"""
        all_splits = self.load_dataset_splits()
        return all_splits.get(self.dataset_name, {})
    
    def _arrow_row_stream(self, folder_path: Path, max_samples: int = None):
        """
        Yield one row at a time from Parquet shards.
        Skips non-Parquet files and supports max_samples limit.
        """
        count = 0

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        for arrow_file in folder_path.glob("*.parquet"): # *.parquet *.[ap][r][c][e][t] *.parquet
            try:
                table = pq.ParquetFile(str(arrow_file))
            except Exception as e:
                print(f"⚠️ Skipping file {arrow_file}: {e}")
                continue

            for batch in table.iter_batches(batch_size=1):
                row = batch.to_pydict()
                row = {k: v[0] for k, v in row.items()}
                
                # --- PIL conversion logic ---
                img_field = row.get("image")
                if isinstance(img_field, dict):
                    # case 1: bytes inside the dict
                    if "bytes" in img_field and img_field["bytes"] is not None:
                        try:
                            row["image"] = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
                        except Exception as e:
                            print(f"[WARN] Failed to convert bytes to image for row {row.get('id')}: {e}")
                    # case 2: path to image file
                    elif "path" in img_field and img_field["path"] is not None:
                        row["image"] = Image.open(img_field["path"]).convert("RGB")
                    else:
                        row["image"] = None
                elif isinstance(img_field, bytes):
                    row["image"] = Image.open(io.BytesIO(img_field)).convert("RGB")
                elif isinstance(img_field, Image.Image):
                    # already PIL, do nothing
                    pass
                else:
                    row["image"] = None  # fallback for unexpected types
                
                yield row

                count += 1
                if max_samples is not None and count >= max_samples:
                    return

"""
FILE_MAPPING template for all leaf-level dataset adapter subclasses.

This mapping standardizes access to dataset files or folders across
different dataset structures (folder-based, JSON-based, or JSON+GT-based).

Each key is a `split_name`, which represents a logical subset of the dataset.
- Use "train", "val", "test" if the dataset has standard splits.
- Use "default" (or "base") if the dataset has no explicit splits.

Each value is a dictionary with the following fields:

    {
        "format": str,           # "folder" | "json" | "json_with_gt"
        "dataset_path": str,     # Path to the folder or JSON file containing dataset samples
        "ground_truth_path": str or None,  # Path to annotation file if needed, else None
        "notes": str or None     # Optional human-readable note about this split
    }

Examples:

1. Folder-per-split dataset (ChartQA):
FILE_MAPPING = {
    "train": {
        "format": "folder",
        "dataset_path": "data/vision/ChartQA/train/",
        "ground_truth_path": None,
        "notes": "Parquet shard files under 'train' folder"
    },
    "val": {
        "format": "folder",
        "dataset_path": "data/vision/ChartQA/val",
        "ground_truth_path": None,
        "notes": "Parquet shard files under 'val' folder"
    },
    "test": {
        "format": "folder",
        "dataset_path": "data/vision/ChartQA/test",
        "ground_truth_path": None,
        "notes": "Parquet shard files under 'test' folder"
    },
}

2. JSON-per-split dataset (DocVQA):
FILE_MAPPING = {
    "train": {
        "format": "json",
        "dataset_path": "data/vision/DocVQA/train.json",
        "ground_truth_path": None,
        "notes": None
    },
    "val": {
        "format": "json",
        "dataset_path": "data/vision/DocVQA/val.json",
        "ground_truth_path": None,
        "notes": None
    },
    "test": {
        "format": "json",
        "dataset_path": "data/vision/DocVQA/test.json",
        "ground_truth_path": None,
        "notes": None
    }
}

Loading logic should dynamically handle:
- 'folder': load all parquet/Parquet files under the folder
- 'json': load JSON as dataset
- 'json_with_gt': load JSON as dataset and merge with ground truth
"""

# ===============================
# OCRDatasetAdapter (Intermediate OCR class)
# ===============================
class OCRDatasetAdapter(BaseDatasetAdapter):
    """
    Intermediate OCR adapter with dataset-agnostic loader helpers
    for folder-based or JSON-based OCR datasets.
    Implements OCR-specific evaluation logic for any OCR dataset.
    """
    def __init__(
            self,
            category: str = "ocr",
            dataset_name: str = "SROIE",
            data_source_from_hf_or_manual: str = "manual",
            hf_repo_name: str = None,
            hf_repo_variant: str = None,
            dataset_split=None
        ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            # hf_repo_name=hf_repo_name,
            # hf_repo_variant=hf_repo_variant
        )
        # Set OCR-specific attributes here directly
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split

    # -----------------------------------------
    # Detection + Recognition Evaluation
    # -----------------------------------------
    def evaluate_detection_recognition(
        self,
        gt_regions,
        pred_regions,
        iou_threshold=0.5
    ):
        matched_gt = set()
        matched_pred = set()
        true_positive = 0

        for i, gt in enumerate(gt_regions):
            for j, pred in enumerate(pred_regions):

                if j in matched_pred:
                    continue

                if not gt["bbox"] or not pred["bbox"]:
                    continue

                iou_score = self._iou(gt["bbox"], pred["bbox"])

                if iou_score >= iou_threshold:
                    gt_text = self._normalize_text(gt["text"])
                    pred_text = self._normalize_text(pred["text"])

                    if gt_text == pred_text:
                        true_positive += 1
                        matched_gt.add(i)
                        matched_pred.add(j)
                        break

        precision = (
            true_positive / len(pred_regions)
            if pred_regions else 0
        )

        recall = (
            true_positive / len(gt_regions)
            if gt_regions else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive
        }
    
    def evaluate_cer(self, gt_regions, pred_regions):
        gt_text = " ".join(
            self._normalize_text(r["text"]) for r in gt_regions
        )

        pred_text = " ".join(
            self._normalize_text(r["text"]) for r in pred_regions
        )

        if len(gt_text) == 0:
            return {"cer": 0}

        distance = self._edit_distance(gt_text, pred_text)
        cer = distance / len(gt_text)

        return {"cer": cer}
    
    def evaluate_wer(self, gt_regions, pred_regions):
        gt_words = []
        for r in gt_regions:
            gt_words.extend(self._normalize_text(r["text"]).split())

        pred_words = []
        for r in pred_regions:
            pred_words.extend(self._normalize_text(r["text"]).split())

        if len(gt_words) == 0:
            return {"wer": 0}

        distance = self._edit_distance(gt_words, pred_words)
        wer = distance / len(gt_words)

        return {"wer": wer}
    
    def evaluate_all_ocr_metrics(self, gt_regions, pred_regions):
        results = {}
        results.update(
            self.evaluate_detection_recognition(gt_regions, pred_regions)
        )
        results.update(self.evaluate_cer(gt_regions, pred_regions))
        results.update(self.evaluate_wer(gt_regions, pred_regions))
        return results
    
    def evaluate_sample(self, gt_regions, pred_regions):
        return self.evaluate_all(gt_regions, pred_regions)

    def evaluate_ocr_regions(self, gt_samples, pred_samples):
        metrics_list = []

        for gt, pred in zip(gt_samples, pred_samples):
            gt_regions = gt.get("regions", [])
            pred_regions = pred.get("regions", []) if isinstance(pred, dict) else []

            metrics = self.evaluate_sample(gt_regions, pred_regions)
            metrics_list.append(metrics)

        if not metrics_list:
            # Hard guarantee: return zeroed metrics instead of crashing
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "cer": 0.0,
                "wer": 0.0
            }

        # Aggregate all metrics
        aggregated = {}

        for key in metrics_list[0].keys():
            aggregated[key] = np.mean([m[key] for m in metrics_list])

        return aggregated
    
    def _normalize_text(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    # -----------------------------
    # Utility: IoU
    # -----------------------------
    def _iou(self, box1: List[int], box2: List[int]) -> float:
        """
        box = [x_min, y_min, x_max, y_max]
        """
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        areaA = (box1[2] - box1[0]) * (box1[3] - box1[1])
        areaB = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = areaA + areaB - inter_area
        iou = inter_area / union if union > 0 else 0
        return iou
    
    def _edit_distance(self, s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )

        return dp[m][n]

    def _hungarian_match(self, gt_regions, pred_regions, iou_threshold=0.5):

        if not gt_regions or not pred_regions:
            return 0, len(gt_regions), len(pred_regions)

        cost_matrix = np.zeros((len(gt_regions), len(pred_regions)))

        for i, gt in enumerate(gt_regions):
            for j, pred in enumerate(pred_regions):
                if gt["bbox"] and pred["bbox"]:
                    cost_matrix[i, j] = 1 - self._iou(gt["bbox"], pred["bbox"])
                else:
                    cost_matrix[i, j] = 1

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        true_positive = 0

        for r, c in zip(row_ind, col_ind):
            iou_score = 1 - cost_matrix[r, c]

            if iou_score >= iou_threshold:
                gt_text = self._normalize_text(gt_regions[r]["text"])
                pred_text = self._normalize_text(pred_regions[c]["text"])

                if gt_text == pred_text:
                    true_positive += 1

        return true_positive, len(gt_regions), len(pred_regions)
    
    # -----------------------------
    # Generic helper: load JSON file
    # -----------------------------
    def _load_json(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------
    # Generic helper: load images from folder
    # -----------------------------
    def _load_image_folder(self, folder_path: Path, exts=("jpg", "png", "jpeg")):
        if not folder_path.exists():
            raise FileNotFoundError(f"Image folder not found: {folder_path}")
        images = []
        for ext in exts:
            images.extend(folder_path.glob(f"*.{ext}"))
        return sorted(images)

    # -----------------------------
    # Generic helper: load images + labels from folders
    # -----------------------------
    def _load_images_and_labels(
        self,
        img_folder: Path,
        box_folder: Path = None,
        entity_folder: Path = None,
        parse_sample_fn=None,
        max_samples_per_split: int = None,
        file_ext: str = ".txt",  # <- added to handle JSON, etc.
    ):
        """
        Generic loader for images + labels from folders.

        img_folder: Path to folder containing images
        box_folder: Path to folder containing OCR boxes (optional)
        entity_folder: Path to folder containing entities (optional)
        parse_sample_fn: function(box_file, entity_file) -> regions list
        max_samples_per_split: max number of samples to load per split
        file_ext: file extension of annotation files (default: .txt)

        Returns:
            List of dicts in universal OCR format
        """
        images = self._load_image_folder(img_folder)
        samples = []

        for i, img_path in enumerate(images):
            if max_samples_per_split is not None and i >= max_samples_per_split:
                break

            regions = []

            if parse_sample_fn:
                box_file = box_folder / f"{img_path.stem}{file_ext}" if box_folder else None
                entity_file = entity_folder / f"{img_path.stem}{file_ext}" if entity_folder else None
                regions = parse_sample_fn(box_file, entity_file)

            samples.append({
                "image": img_path,
                "regions": regions
            })

        return samples

    # -----------------------------
    # OCR dataset loader (central entry)
    # -----------------------------
    def ocr_dataset_loader(
        self,
        split: str,
        file_mapping: dict,
        parse_sample_fn=None,
        max_samples_per_split: int = None
    ):
        """
        Central loader for any OCR dataset using its FILE_MAPPING
        and parsing function.

        Args:
            split: "train" | "test"
            file_mapping: FILE_MAPPING dict
            parse_sample_fn: function to parse a single sample into regions
            max_samples_per_split: limit number of samples

        Returns:
            List of samples in universal OCR format
        """
        if split not in list(file_mapping.keys()):
            raise ValueError(f"Split '{split}' not in FILE_MAPPING")

        split_info = file_mapping[split]
        dataset_path = Path(split_info["dataset_path"])

        # Folder-based loader
        if split_info["format"] == "folder":
            # For FUNSD/SROIE: box_folder and entity_folder optional
            img_folder = dataset_path / "images" if (dataset_path / "images").exists() else dataset_path / "img"
            box_folder = dataset_path / "box" if (dataset_path / "box").exists() else None
            entity_folder = dataset_path / "entities" if (dataset_path / "entities").exists() else None

            return self._load_images_and_labels(
                img_folder=img_folder,
                box_folder=box_folder,
                entity_folder=entity_folder,
                parse_sample_fn=parse_sample_fn,
                max_samples_per_split=max_samples_per_split
            )

        # JSON-based loader (future proof)
        elif split_info["format"] in ("json", "json_with_gt"):
            return self._load_json(dataset_path)

        else:
            raise NotImplementedError(f"Unsupported dataset format: {split_info['format']}")
    
    def _arrow_row_to_token_ocr(self, row):
        """
        Extract token-level OCR info from Parquet row.
        Works for SROIE, FUNSD, and similar datasets.
        """
        words = row.get("words", [])
        bboxes = row.get("bboxes", [])

        if not words or not bboxes:
            return None

        for i, box in enumerate(bboxes):
            print(f"DEBUG box[{i}]:", box, type(box))

        # Normalize bboxes if needed
        norm_bboxes = []
        for box in bboxes:
            if isinstance(box, (list, tuple)):
                if len(box) == 4:
                    norm_bboxes.append(box)
                elif len(box) == 8:
                    xs = box[0::2]
                    ys = box[1::2]
                    norm_bboxes.append(
                        [min(xs), min(ys), max(xs), max(ys)]
                    )
                else:
                    # fallback to zero-box
                    norm_bboxes.append([0, 0, 0, 0])
            else:
                norm_bboxes.append([0, 0, 0, 0])

        # If words and bboxes count mismatch, truncate to shorter length
        n_tokens = min(len(words), len(norm_bboxes))
        tokens = [{"text": str(words[i]), "bbox": norm_bboxes[i]} for i in range(n_tokens)]
        return tokens
    
    def _build_universal_ocr_sample(
        self,
        image,
        ocr_tokens,
        split: str,
        sample_id: str,
        token_labels=None,
        document_entities=None,
    ):
        # words = ocr_tokens.get("words", []) if isinstance(ocr_tokens, dict) else [t["text"] for t in ocr_tokens]
        # bboxes = ocr_tokens.get("bboxes", []) if isinstance(ocr_tokens, dict) else [t["bbox"] for t in ocr_tokens]

        words = [t["text"] for t in ocr_tokens]
        bboxes = [t["bbox"] for t in ocr_tokens]

        for idx, (text, b) in enumerate(zip(words, bboxes)):
            print(f"[DEBUG OCR TOKEN] {idx}: {b} -> '{text}'")

        return {
            "input": {
                "image": image,
                "ocr": {
                    "words": words,
                    "bboxes": bboxes
                }
            },
            "ground_truth": {
                "token_labels": token_labels,          # FUNSD only
                "document_entities": document_entities # SROIE only
            },
            "metadata": {
                "dataset": self.dataset_name,
                "split": split,
                "sample_id": sample_id
            }
        }

        # return {
        #     "input": {
        #         "image": image,
        #         "ocr": {
        #             "words": words,
        #             "bboxes": bboxes
        #         }
        #     },
        #     "ground_truth": {
        #         "token_labels": token_labels or [],
        #         "document_entities": document_entities
        #     },
        #     "metadata": {
        #         "dataset": self.category,
        #         "split": split,
        #         "sample_id": sample_id
        #     }
        # }
    
    def extract_sroie_entities(self, words):
        text = " ".join(words)
        text_upper = text.upper()

        entities = {}

        # ---------- COMPANY ----------
        company_candidates = []
        for w in words:
            w_up = w.upper()
            if any(k in w_up for k in ["SDN", "BHD", "BND"]):
                company_candidates.append(w)

        if company_candidates:
            company = " ".join(company_candidates)
            company = company.replace("BND", "BHD")
            company = company.replace("(", " (")
            company = " ".join(company.split())
            entities["company"] = company

        # ---------- DATE ----------
        date_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text)
        if date_match:
            entities["date"] = date_match.group(0)

        # ---------- TOTAL ----------
        total_candidates = []
        for w in words:
            w_clean = w.replace("RM", "").strip()
            if re.fullmatch(r"\d+\.\d{2}", w_clean):
                total_candidates.append(w_clean)

        if total_candidates:
            # Heuristic: highest monetary value is usually TOTAL
            entities["total"] = sorted(
                total_candidates,
                key=lambda x: float(x),
                reverse=True
            )[0]

        print("[DEBUG extracted entities]", entities)
        return entities
    
    def evaluate_token_ner(self, gt_labels, pred_labels):
        """
        Simple token-level NER accuracy.
        """
        if not gt_labels or not pred_labels:
            return {"accuracy": 0.0}

        n = min(len(gt_labels), len(pred_labels))
        correct = sum(
            1 for i in range(n)
            if gt_labels[i] == pred_labels[i]
        )

        return {"accuracy": correct / n}

    def evaluate_document_kie(self, gt_entities, pred_entities):
        """
        Exact-match document-level KIE evaluation.
        """
        if not gt_entities:
            return {"f1": 0.0}

        tp = sum(
            1 for k, v in gt_entities.items()
            if pred_entities.get(k) == v
        )

        precision = tp / len(pred_entities) if pred_entities else 0
        recall = tp / len(gt_entities)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    def character_error_rate(self, pred, gt):
        """
        CER = edit_distance / len(gt)
        """
        if not pred or not gt:
            return 1.0

        sm = SequenceMatcher(None, pred, gt)
        edits = sum(triple.size for triple in sm.get_matching_blocks())
        return 1.0 - edits / max(len(gt), 1)
    
    def soft_entity_match(self, pred_entities, gt_entities, cer_threshold=0.3):
        """
        Returns precision, recall, f1 using CER-based soft matching
        """
        tp = 0
        for k, gt_val in gt_entities.items():
            pred_val = pred_entities.get(k)
            if not pred_val:
                continue

            cer = self.character_error_rate(pred_val.lower(), gt_val.lower())
            if cer <= cer_threshold:
                tp += 1

        precision = tp / max(len(pred_entities), 1)
        recall = tp / max(len(gt_entities), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        return precision, recall, f1

    def _arrow_folder_samples(
        self,
        folder_path: Path,
        split: str,
        max_samples_per_split=None
    ):
        """
        Generator yielding universal OCR samples from Parquet folder
        """
        for idx, row in enumerate(self._arrow_row_stream(folder_path, max_samples_per_split)):
            ocr_tokens = self._arrow_row_to_token_ocr(row)
            if ocr_tokens is None:
                continue

            sample_id = row.get("id") or row.get("key") or f"{split}_{idx}"
            yield self._build_universal_ocr_sample(
                image=row.get("image"),  # lazy-loaded PIL object
                ocr_tokens=ocr_tokens,
                token_labels=row.get("ner_tags"),  # FUNSD only
                document_entities=row.get("entities"),  # SROIE only
                split=split,
                sample_id=sample_id
            )

# ------------------------
# OCR Adapters
# ------------------------
"""
universal_format_for_OCR_datasets = {
    "input": {
        "image": PIL.Image,

        # Always token-aligned OCR
        "ocr": {
            "words": List[str],
            "bboxes": List[List[int]]  # [x1, y1, x2, y2]
        }
    },

    "ground_truth": {
        # Token-level labels (FUNSD)
        # None for datasets like SROIE
        "token_labels": Optional[List[str]],

        # Document-level labels (SROIE)
        # None for datasets like FUNSD
        "document_entities": Optional[Dict[str, str]]
    },

    "metadata": {
        "dataset": str,      # "SROIE" | "FUNSD"
        "split": str,        # "train" | "test"
        "sample_id": str     # key / id / index
    }
}

BaseDatasetAdapter
- Load Parquet shards
- Return raw Parquet rows

SROIEAdapter / FUNSDAdapter (leaf adapters)
- Interpret Parquet schema
- Convert Parquet row → universal OCR format

OCRDatasetAdapter
- common functions shared by SROIEAdapter / FUNSDAdapter (leaf adapters)

eval_postprocess_utils.py
- Evaluate OCR outputs after conversion, against ground truth
- Provide generic OCR metrics
"""
class SROIEAdapter(OCRDatasetAdapter):
    """
    data/ocr/SROIE/train/data-00000-of-00001.parquet,
    is the only parquet shard file in train/ sub-folder.
    
    === Dataset object ===
    Dataset({
        features: ['image', 'key', 'image_size', 'entities', 'words', 'bboxes'],
        num_rows: 626
    })

    === Dataset features ===
    {'bboxes': List(List(Value('int64'))),
    'entities': {'address': Value('string'),
                'company': Value('string'),
                'date': Value('string'),
                'total': Value('string')},
    'image': Image(mode=None, decode=True),
    'image_size': {'height': Value('int64'), 'width': Value('int64')},
    'key': Value('string'),
    'words': List(Value('string'))}

    === First 1 sample ===
    {'bboxes': [[72, 25, 326, 64],
                [50, 82, 440, 121],
                [205, 121, 285, 139],
                [110, 144, 383, 163],
                [192, 169, 299, 187],
                [162, 193, 334, 211],
                [217, 216, 275, 233],
                [50, 342, 279, 359],
                [50, 372, 96, 390],
                [165, 372, 342, 389],
                [48, 396, 117, 415],
                [164, 397, 215, 413],
                [49, 423, 122, 440],
                [191, 460, 298, 476],
                [30, 508, 121, 523],
                [200, 507, 247, 521],
                [276, 506, 306, 522],
                [374, 507, 441, 521],
                [69, 531, 102, 550],
                [221, 531, 247, 545],
                [420, 529, 443, 547],
                [27, 570, 137, 583],
                [159, 570, 396, 584],
                [77, 598, 113, 613],
                [138, 597, 148, 607],
                [202, 597, 245, 612],
                [275, 598, 309, 612],
                [411, 596, 443, 613],
                [245, 639, 293, 658],
                [118, 671, 291, 687],
                [408, 669, 443, 684],
                [86, 704, 292, 723],
                [401, 703, 443, 719],
                [205, 744, 243, 765],
                [402, 748, 441, 763],
                [205, 770, 271, 788],
                [412, 772, 443, 786],
                [97, 845, 401, 860],
                [190, 864, 309, 880],
                [142, 883, 353, 901],
                [137, 903, 351, 920],
                [202, 942, 292, 959],
                [163, 962, 330, 977],
                [412, 639, 442, 654]],
    'entities': {'address': 'NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 '
                            'JOHOR BAHRU, JOHOR.',
                'company': 'BOOK TA .K (TAMAN DAYA) SDN BHD',
                'date': '25/12/2018',
                'total': '9.00'},
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=463x1013 at 0x2197EF66CD0>,
    'image_size': {'height': 1013, 'width': 463},
    'key': 'X00016469612',
    'words': ['TAN WOON YANN',
            'BOOK TA .K(TAMAN DAYA) SDN BND',
            '789417-W',
            'NO.53 55',
            'TAMAN DAYA',
            '81100 JOHOR BAHRU',
            'JOHOR.',
            'DOCUMENT NO : TD01167104',
            'DATE:',
            '25/12/2018 8:13:39 PM',
            'CASHIER:',
            'MANIS',
            'MEMBER:',
            'CASH BILL',
            'CODE/DESC',
            'PRICE',
            'DISC',
            'AMOUNT',
            'QTY',
            'RM',
            'RM',
            '9556939040116',
            'KF MODELLING CLAY KIDDY FISH',
            '1 PC',
            '*',
            '9.000',
            '0.00',
            '9.00',
            'TOTAL:',
            'ROUR DING ADJUSTMENT:',
            '0.00',
            'ROUND D TOTAL (RM):',
            '9.00',
            'CASH',
            '10.00',
            'CHANGE',
            '1.00',
            'GOODS SOLD ARE NOT RETURNABLE OR',
            'EXCHANGEABLE',
            '***',
            '***',
            'THANK YOU',
            'PLEASE COME AGAIN !',
            '9.00']}

    data/ocr/SROIE/test/data-00000-of-00001.parquet,
    is the only parquet shard files in test/ sub-folder.

    The sub-folder structure is consistent across train and test split:
    data/ocr/SROIE/test/
    data/ocr/SROIE/train/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under test/, and train/ folders.
    as well as as the pathing down to the individual sample level.
    The parsing logic for each sample is implemented in the _parse_sroie_sample() function, 
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """

    FILE_MAPPING = {
        "train": {
            "format": "folder",
            "dataset_path": "data/ocr/SROIE/train",
            "ground_truth_path": None,
            "notes": "Train folder containing 1 Parquet shard file"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/ocr/SROIE/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Parquet shard file"
        }
    }

    def __init__(
        self,
        category: str = "ocr",
        dataset_name: str = "SROIE",
        data_source_from_hf_or_manual: str = "manual",
        hf_repo_name: str = "jsdnrs/ICDAR2019-SROIE",
        hf_repo_variant: str = None,
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING # <- assign class-level mapping to instance

        
    def load_split(
        self,
        dataset_split=None,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
        **kwargs,
    ):
        """Load one split or all splits if dataset_split is None."""
        splits_to_load = (
            [dataset_split or self.dataset_split]
            if (dataset_split or self.dataset_split)
            else list(self.FILE_MAPPING.keys())
        )

        all_samples = []

        for split in splits_to_load:
            if split not in self.FILE_MAPPING:
                raise ValueError(
                    f"SROIE supports splits: {list(self.FILE_MAPPING.keys())}"
                )

            split_path = Path().cwd() / self.FILE_MAPPING[split]["dataset_path"]
            use_hf = self.data_source_from_hf_or_manual == "hf" or not split_path.exists()

            if use_hf:
                # Load from HuggingFace when repo is HF or local path missing (e.g. notebook/Colab)
                ds = load_dataset(self.hf_repo_name, split=split)
                n = min(len(ds), max_samples_per_split or len(ds))
                for idx in range(n):
                    row = ds[idx]
                    img = row.get("image")
                    if img is None:
                        continue
                    if hasattr(img, "convert"):
                        img = img.convert("RGB")
                    row = {"image": img, "key": row.get("key", f"{split}_{idx}"), "entities": row.get("entities"), "words": row.get("words", []), "bboxes": row.get("bboxes", [])}
                    ocr_tokens = self._arrow_row_to_token_ocr(row)
                    if ocr_tokens is None:
                        continue
                    sample = self._build_universal_ocr_sample(
                        image=row.get("image"),
                        ocr_tokens=ocr_tokens,
                        token_labels=None,
                        document_entities=row.get("entities"),
                        split=split,
                        sample_id=row.get("key", f"{split}_{idx}")
                    )
                    all_samples.append(sample)
            else:
                # Load Parquet shards from local path
                rows = self._arrow_row_stream(split_path, max_samples=max_samples_per_split)
                for idx, row in enumerate(rows):
                    img_field = row.get("image")
                    if isinstance(img_field, bytes):
                        row["image"] = Image.open(io.BytesIO(img_field)).convert("RGB")
                    ocr_tokens = self._arrow_row_to_token_ocr(row)
                    if ocr_tokens is None:
                        continue
                    sample = self._build_universal_ocr_sample(
                        image=row.get("image"),
                        ocr_tokens=ocr_tokens,
                        token_labels=None,
                        document_entities=row.get("entities"),
                        split=split,
                        sample_id=row.get("key", f"{split}_{idx}")
                    )
                    all_samples.append(sample)

        if max_samples_per_category:
            all_samples = all_samples[:max_samples_per_category]

        return all_samples

class FUNSDAdapter(OCRDatasetAdapter):
    """
    data/ocr/FUNSD/train/data-00000-of-00001.parquet,
    is the only parquet shard file in train/ sub-folder.
    
    === Dataset object ===
    Dataset({
        features: ['id', 'words', 'bboxes', 'ner_tags', 'image'],
        num_rows: 149
    })

    === Dataset features ===
    {'bboxes': List(List(Value('int64'))),
    'id': Value('string'),
    'image': Image(mode=None, decode=True),
    'ner_tags': List(ClassLabel(names=['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER'])),
    'words': List(Value('string'))}
    
    data/ocr/FUNSD/test/data-00000-of-00001.parquet,
    is the only parquet shard files in test/ sub-folder.

    The sub-folder structure is consistent across train and test split:
    data/ocr/FUNSD/test/
    data/ocr/FUNSD/train/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under test/, and train/ folders.
    as well as as the pathing down to the individual sample level.
    The parsing logic for each sample is implemented in the _parse_funsd_annotation() function, 
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    FILE_MAPPING = {
        "train": {
            "format": "folder",
            "dataset_path": "data/ocr/FUNSD/train",
            "ground_truth_path": None,
            "notes": "Train folder containing 1 Parquet shard file"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/ocr/FUNSD/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Parquet shard file"
        }
    }

    def __init__(
        self,
        category: str = "ocr",
        dataset_name: str = "FUNSD",
        data_source_from_hf_or_manual: str = "manual",
        hf_repo_name: str = "nielsr/funsd",
        hf_repo_variant: str = None,
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING
        self.id2label = {
            0: "O",
            1: "B-HEADER",
            2: "I-HEADER",
            3: "B-QUESTION",
            4: "I-QUESTION",
            5: "B-ANSWER",
            6: "I-ANSWER",
        }

    def compute_ner_f1(self, pred_labels, gold_labels, mask_o=True):
        """
        Token-level NER evaluation.
        Returns micro + macro precision/recall/F1.
        """

        assert len(pred_labels) == len(gold_labels)

        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        label_set = set()

        for pred, gold in zip(pred_labels, gold_labels):
            if mask_o and gold == "O":
                continue

            label_set.add(gold)
            label_set.add(pred)

            if pred == gold:
                tp[gold] += 1
            else:
                fp[pred] += 1
                fn[gold] += 1

        # ---- micro ----
        micro_tp = sum(tp.values())
        micro_fp = sum(fp.values())
        micro_fn = sum(fn.values())

        micro_precision = micro_tp / (micro_tp + micro_fp + 1e-9)
        micro_recall = micro_tp / (micro_tp + micro_fn + 1e-9)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-9)

        # ---- macro ----
        per_label_f1 = []
        for label in label_set:
            if mask_o and label == "O":
                continue

            l_tp = tp[label]
            l_fp = fp[label]
            l_fn = fn[label]

            p = l_tp / (l_tp + l_fp + 1e-9)
            r = l_tp / (l_tp + l_fn + 1e-9)
            f1 = 2 * p * r / (p + r + 1e-9)

            per_label_f1.append(f1)

        macro_f1 = sum(per_label_f1) / max(len(per_label_f1), 1)

        return {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1
        }

    def evaluate_sample(self, sample, prediction):
        gold_ids = sample.get("token_labels", [])
        pred_ids = prediction.get("token_labels", [])

        if not gold_ids or not pred_ids:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "macro_f1": 0.0
            }

        gold_labels = [self.id2label[i] for i in gold_ids]
        pred_labels = [self.id2label[i] for i in pred_ids]

        metrics = self.compute_ner_f1(
            pred_labels=pred_labels,
            gold_labels=gold_labels,
            mask_o=True
        )

        return {
            "precision": metrics["micro_precision"],
            "recall": metrics["micro_recall"],
            "f1": metrics["micro_f1"],
            "macro_f1": metrics["macro_f1"]
        }
    
    def evaluate_ocr_regions(self, *args, **kwargs):
        return None

    def load_split(self, 
        dataset_split=None, 
        max_samples_per_split=None, 
        max_samples_per_category=None,
        only_splits_with_gt=False,
        **kwargs,
    ):
        """
        Load one split or all splits if dataset_split is None.

        FUNSD is Parquet-based:
        - Each split folder contains one or more .parquet shard files
        - Parsing is row-based and handled by OCRDatasetAdapter helpers
        """
        splits_to_load = (
            [dataset_split or self.dataset_split]
            if (dataset_split or self.dataset_split)
            else list(self.FILE_MAPPING.keys())
        )
        print(f"splits_to_load: {splits_to_load}")

        all_samples = []

        for split in splits_to_load:
            if split not in self.FILE_MAPPING:
                raise ValueError(f"FUNSD supports splits: {list(self.FILE_MAPPING.keys())}")

            split_info = self.FILE_MAPPING[split]
            split_dir = Path().cwd() / split_info["dataset_path"]
            use_hf = self.data_source_from_hf_or_manual == "hf" or not split_dir.exists()

            if use_hf:
                ds = load_dataset(self.hf_repo_name, split=split)
                n = min(len(ds), max_samples_per_split or len(ds))
                for idx in range(n):
                    row = ds[idx]
                    img = row.get("image")
                    if img is None:
                        continue
                    if hasattr(img, "convert"):
                        img = img.convert("RGB")
                    row = {"image": img, "id": row.get("id", str(idx)), "words": row.get("words", []), "bboxes": row.get("bboxes", []), "ner_tags": row.get("ner_tags")}
                    ocr_tokens = self._arrow_row_to_token_ocr(row)
                    if ocr_tokens is None:
                        continue
                    sample = self._build_universal_ocr_sample(
                        image=row.get("image"),
                        ocr_tokens=ocr_tokens,
                        token_labels=row.get("ner_tags"),
                        document_entities=None,
                        split=split,
                        sample_id=row.get("id", f"{split}_{idx}")
                    )
                    all_samples.append(sample)
            else:
                for sample in self._arrow_folder_samples(split_dir, split, max_samples_per_split):
                    all_samples.append(sample)

            # # Load Parquet shards
            # rows = self._arrow_row_stream(split_dir, max_samples=max_samples_per_split)
            # split_samples = []

            # for idx, row in enumerate(rows):
            #     # --- PIL in-memory conversion ---
            #     img_field = row.get("image")
            #     if isinstance(img_field, bytes):
            #         row["image"] = Image.open(io.BytesIO(img_field)).convert("RGB")

            #     ocr_tokens = self._arrow_row_to_token_ocr(row)
            #     sample = self._build_universal_ocr_sample(
            #         # row=row,
            #         # dataset_name=self.dataset_name,
            #         # split=split,
            #         # task_type="document_kie",  

            #         image=row.get("image"), # PIL Image object
            #         ocr_tokens=ocr_tokens,
            #         token_labels=row.get("ner_tags"), # FUNSD supports NER + document KIE
            #         document_entities=None,
            #         split=split,
            #         sample_id=row.get("id", f"{split}_{idx}")
            #     )
            #     split_samples.append(sample)

            #     if max_samples_per_split and len(split_samples) >= max_samples_per_split:
            #         break

            # all_samples.extend(split_samples)

        if max_samples_per_category:
            all_samples = all_samples[:max_samples_per_category]

        return all_samples

# ===============================
# VisionDatasetAdapter (Intermediate Vision class)
# ===============================
class VisionDatasetAdapter(BaseDatasetAdapter):
    """
    Intermediate Vision adapter with dataset-agnostic loader helpers
    for folder-based (Parquet) or JSON-based Vision datasets.

    All Vision datasets must return a unified sample schema:

    {
        "image": image_path_or_bytes_or_id,
        "question": str or None,
        "answer": str or None,
        "metadata": dict (optional)
    }
    """
    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "DocVQA",
        data_source_from_hf_or_manual: str = "manual",
        hf_repo_name: str = None,
        hf_repo_variant: str = None,
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant

    # -----------------------------------------
    # Generic helper: load JSON file
    # -----------------------------------------
    def _load_json(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------------------
    # Generic helper: merge dataset JSON + GT JSON
    # -----------------------------------------
    def _merge_json_with_gt(self, dataset_json, gt_json):
        """
        Merge DUDE-style dataset + ground truth.
        Assumes both are list-like and aligned by index or ID.
        """
        merged = []

        # If list-based
        if isinstance(dataset_json, list) and isinstance(gt_json, list):
            for sample, gt in zip(dataset_json, gt_json):
                merged_sample = sample.copy()
                merged_sample["answer"] = gt.get("answer")
                merged.append(merged_sample)
            return merged

        # If dict-based keyed by ID
        if isinstance(dataset_json, dict) and isinstance(gt_json, dict):
            for key in dataset_json:
                merged_sample = dataset_json[key].copy()
                if key in gt_json:
                    merged_sample["answer"] = gt_json[key].get("answer")
                merged.append(merged_sample)
            return merged

        raise ValueError("Unsupported JSON + GT structure")

    # -----------------------------------------
    # Vision dataset loader (central entry)
    # -----------------------------------------
    def vision_dataset_loader(
        self,
        split: str,
        file_mapping: dict,
        parse_sample_fn=None,
        max_samples_per_split: int = None,
        max_samples_per_category: int = None
    ):
        """
        Central loader for Vision datasets using FILE_MAPPING.

        Supports:
            - folder (Parquet shards)
            - json
            - json_with_gt
        """

        if split not in file_mapping.keys():
            raise ValueError(f"Split '{split}' not in FILE_MAPPING")

        split_info = file_mapping[split]
        dataset_path = Path().cwd() / split_info["dataset_path"]

        fmt = split_info["format"]

        # -------------------------
        # Folder-based (Parquet)
        # -------------------------
        if fmt == "folder":
            samples = self._load_arrow_folder(
                dataset_path,
                max_samples_per_split=max_samples_per_split
            )
            return samples

        # -------------------------
        # JSON-only
        # -------------------------
        elif fmt == "json":
            if dataset_path.is_dir():
                # Some datasets keep parquet shards under split directories while FILE_MAPPING uses "json".
                return self._load_arrow_folder(
                    dataset_path,
                    max_samples_per_split=max_samples_per_split,
                    max_samples_per_category=max_samples_per_category,
                )

            data = self._load_json(dataset_path)

            # If JSON has top-level key like {"data": [...]}
            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                elif "questions" in data:
                    data = data["questions"]

            if max_samples_per_split:
                data = data[:max_samples_per_split]

            return data

        # -------------------------
        # JSON + GT (DUDE-style)
        # -------------------------
        elif fmt == "json_with_gt":
            dataset_json = self._load_json(dataset_path)

            gt_path = Path() / split_info["ground_truth_path"]
            gt_json = self._load_json(gt_path)

            merged = self._merge_json_with_gt(dataset_json, gt_json)

            if max_samples_per_split:
                merged = merged[:max_samples_per_split]

            return merged

        else:
            raise NotImplementedError(
                f"Unsupported dataset format: {fmt}"
            )

# ------------------------
# Vision Adapters
# ------------------------
"""
universal_vision_sample = {
    "input": {  # everything that serves as model input
        "image": <PIL.Image.Image> or List[<PIL.Image.Image>],  # single image or multiple images (e.g., MMMU datasets)
        "question": str or None,  # question/prompt associated with the image, if available
        "other_modalities": dict  # optional additional fields, e.g., OCR, reasoning info
    },
    "ground_truth": Any,  # label/answer, could be:
        # - string (DocVQA, InfographicsVQA, MMMU datasets)
        # - list of strings (ChartQA)
    "metadata": {
        "dataset": str,  # dataset name, e.g., "DocVQA"
        "split": str,  # split name, e.g., "test", "train", "validation"
        "sample_id": int,  # zero-based index
        "additional_info": dict  # optional, e.g., docId, ucsf_document_id, human_or_machine, page_info, subfield, img_type
    }
}
"""
class DocVQAAdapter(VisionDatasetAdapter):
    """
    data/vision/DocVQA/test/data-00000-of-00008.parquet
    
    === Dataset object ===
    Dataset({
        features: ['questionId', 'question', 'question_types', 'image', 'docId', 'ucsf_document_id', 'ucsf_document_page_no', 'answers', 'data_split'],
        num_rows: 649
    })

    === Dataset features ===
    {'answers': List(Value('string')),
    'data_split': Value('string'),
    'docId': Value('int64'),
    'image': Image(mode=None, decode=True),
    'question': Value('string'),
    'questionId': Value('string'),
    'question_types': List(Value('string')),
    'ucsf_document_id': Value('string'),
    'ucsf_document_page_no': Value('string')}

    === First 1 sample ===
    {'answers': None,
    'data_split': 'test',
    'docId': 4720,
    'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=1653x2339 at 0x20A409A6850>,
    'question': 'What is the dividend payout in 2012?',
    'questionId': '57344',
    'question_types': None,
    'ucsf_document_id': 'rnbx0223',
    'ucsf_document_page_no': '193'}

    data/vision/DocVQA/test/data-00001-of-00008.parquet,
    data/vision/DocVQA/test/data-00002-of-00008.parquet,
    data/vision/DocVQA/test/data-00003-of-00008.parquet,
    data/vision/DocVQA/test/data-00004-of-00008.parquet,
    data/vision/DocVQA/test/data-00005-of-00008.parquet,
    data/vision/DocVQA/test/data-00006-of-00008.parquet,
    data/vision/DocVQA/test/data-00007-of-00008.parquet,
    are the next remaining parquet shard files in the same test/ sub-folder.

    data/vision/DocVQA/validation/data-00000-of-00008.parquet,
    data/vision/DocVQA/validation/data-00001-of-00008.parquet,
    data/vision/DocVQA/validation/data-00002-of-00008.parquet,
    data/vision/DocVQA/validation/data-00003-of-00008.parquet,
    data/vision/DocVQA/validation/data-00004-of-00008.parquet,
    data/vision/DocVQA/validation/data-00005-of-00008.parquet,
    data/vision/DocVQA/validation/data-00006-of-00008.parquet,
    data/vision/DocVQA/validation/data-00007-of-00008.parquet,
    are the parquet shard files in the validation/ sub-folder.

    The sub-folder structure is consistent across test and validation splits:
    data/vision/DocVQA/test/
    data/vision/DocVQA/validation/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under test/ and validation/ folders.
    The parsing logic for each sample is implemented in the _parse_docvqa_annotation() function,
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    # Only validation has ground truth in the dataset; test has answers: null.
    SPLITS_WITH_GT = {"validation"}

    FILE_MAPPING = {
        # "test": {  # no GT (answers: null); evaluation uses validation only
        #     "format": "folder",
        #     "dataset_path": "data/vision/DocVQA/test",
        #     "ground_truth_path": None,
        #     "notes": "Test folder containing 8 Parquet shard files"
        # },
        "validation": {
            "format": "json",
            "dataset_path": "data/vision/DocVQA/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 8 Parquet shard files"
        },
    }

    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "DocVQA",
        data_source_from_hf_or_manual: str = "manual",
        hf_repo_name: str = "lmms-lab/DocVQA",
        hf_repo_variant: str = "DocVQA",
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING
    
    def load_split(
        self,
        dataset_split=None,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
    ):
        """Stream DocVQA samples in universal vision format, one row at a time."""
        all_splits = list(self.FILE_MAPPING.keys())
        if dataset_split or self.dataset_split:
            splits_to_load = [dataset_split or self.dataset_split]
        else:
            splits_to_load = all_splits
        if only_splits_with_gt and getattr(self, "SPLITS_WITH_GT", None):
            splits_to_load = [s for s in splits_to_load if s in self.SPLITS_WITH_GT]

        def _iter():
            emitted = 0
            for split in splits_to_load:
                raw_samples = self.vision_dataset_loader(
                    split,
                    self.FILE_MAPPING,
                    max_samples_per_split=max_samples_per_split,
                )

                # standardize to universal_format_for_vision_datasets
                for idx, s in enumerate(raw_samples):
                    yield {
                        "input": {
                            "image": s.get("image"),
                            "question": s.get("question"),
                        },
                        "ground_truth": s.get("answers")[0] if s.get("answers") else None,
                        "metadata": {
                            "dataset": "DocVQA",
                            "split": split,
                            "sample_id": f"{split}_{idx}",
                        },
                    }
                    emitted += 1
                    if max_samples_per_category and emitted >= max_samples_per_category:
                        return

        return _iter()

class InfographicsVQAAdapter(VisionDatasetAdapter):
    """
    data/vision/InfographicsVQA/test/data-00000-of-00004.parquet
    
    === Dataset object ===
    Dataset({
        features: ['questionId', 'question', 'answers', 'answer_type', 'image', 'image_url', 'operation/reasoning', 'ocr', 'data_split'],
        num_rows: 822
    })

    === Dataset features ===
    {'answer_type': List(Value('string')),
    'answers': List(Value('string')),
    'data_split': Value('string'),
    'image': Image(mode=None, decode=True),
    'image_url': Value('string'),
    'ocr': Value('string'),
    'operation/reasoning': List(Value('string')),
    'question': Value('string'),
    'questionId': Value('string')}
    
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_infographicsvqa_annotation().
    
    data/vision/InfographicsVQA/test/data-00001-of-00004.parquet,
    data/vision/InfographicsVQA/test/data-00002-of-00004.parquet,
    data/vision/InfographicsVQA/test/data-00003-of-00004.parquet,
    are the next remaining parquet shard files in the same test/ sub-folder.

    data/vision/InfographicsVQA/validation/data-00000-of-00004.parquet,
    data/vision/InfographicsVQA/validation/data-00001-of-00004.parquet,
    data/vision/InfographicsVQA/validation/data-00002-of-00004.parquet,
    data/vision/InfographicsVQA/validation/data-00003-of-00004.parquet,
    are the parquet shard files in the validation/ sub-folder.

    The sub-folder structure is consistent across test and validation splits:
    data/vision/InfographicsVQA/test/
    data/vision/InfographicsVQA/validation/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under test/ and validation/ folders.
    The parsing logic for each sample is implemented in the _parse_infographicsvqa_annotation() function,
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    # Only validation has ground truth; test may have null/unreliable labels.
    SPLITS_WITH_GT = {"validation"}

    FILE_MAPPING = {
        # "test": {  # no GT for evaluation; evaluation uses validation only
        #     "format": "folder",
        #     "dataset_path": "data/vision/InfographicsVQA/test",
        #     "ground_truth_path": None,
        #     "notes": "Test folder containing 4 Parquet shard files"
        # },
        "validation": {
            "format": "folder",
            "dataset_path": "data/vision/InfographicsVQA/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 4 Parquet shard files"
        },
    }


    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "InfographicsVQA",
        data_source_from_hf_or_manual: str = "manual",
        hf_repo_name: str = "lmms-lab/DocVQA",
        hf_repo_variant: str = "InfographicVQA",
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING
    
    def load_split(
        self,
        dataset_split=None,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
    ):
        """Stream InfographicsVQA samples in universal vision format."""
        all_splits = list(self.FILE_MAPPING.keys())
        if dataset_split or self.dataset_split:
            splits_to_load = [dataset_split or self.dataset_split]
        else:
            splits_to_load = all_splits
        if only_splits_with_gt and getattr(self, "SPLITS_WITH_GT", None):
            splits_to_load = [s for s in splits_to_load if s in self.SPLITS_WITH_GT]

        def _iter():
            emitted = 0
            for split in splits_to_load:
                raw_samples = self.vision_dataset_loader(
                    split,
                    self.FILE_MAPPING,
                    max_samples_per_split=max_samples_per_split,
                )

                for idx, s in enumerate(raw_samples):
                    yield {
                        "input": {
                            "image": s.get("image"),
                            "question": s.get("question"),
                        },
                        "ground_truth": s.get("answers")[0] if s.get("answers") else None,
                        "metadata": {
                            "dataset": "InfographicsVQA",
                            "split": split,
                            "sample_id": f"{split}_{idx}",
                        },
                    }
                    emitted += 1
                    if max_samples_per_category and emitted >= max_samples_per_category:
                        return

        return _iter()

class ChartQAAdapter(VisionDatasetAdapter):
    """
    data/vision/ChartQA/test/data-00000-of-00001.parquet,
    is the only parquet shard file in test/ sub-folder.
    
    === Dataset object ===
    Dataset({
        features: ['image', 'query', 'label', 'human_or_machine'],
        num_rows: 2500
    })

    === Dataset features ===
    {'human_or_machine': ClassLabel(names=['human', 'machine']),
    'image': Image(mode=None, decode=True),
    'label': List(Value('string')),
    'query': Value('string')}

    === First 2 samples ===
    --- Sample 0 ---
    {'human_or_machine': 0,
    'image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=850x600 at 0x1DA8E53BE50>,
    'label': ['14'],
    'query': 'How many food item is shown in the bar graph?'}

    --- Sample 1 ---
    {'human_or_machine': 0,
    'image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=850x600 at 0x1DA4DE0ED90>,
    'label': ['0.57'],
    'query': 'What is the difference in value between Lamb and Corn?'}


    we need to load it once and print out the first 3 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_chartqa_annotation().
    
    data/vision/ChartQA/train/data-00000-of-00003.parquet,
    data/vision/ChartQA/train/data-00001-of-00003.parquet,
    data/vision/ChartQA/train/data-00002-of-00003.parquet,
    are the 3 parquet shard files in train/ sub-folder.

    data/vision/ChartQA/val/data-00000-of-00001.parquet,
    is the only parquet shard file in val/ sub-folder.

    The sub-folder structure is consistent across test, train, and val splits:
    data/vision/ChartQA/test/
    data/vision/ChartQA/train/
    data/vision/ChartQA/val/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under test/, train/, and val/ folders.
    The parsing logic for each sample is implemented in the _parse_chartqa_annotation() function,
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    FILE_MAPPING = {
        "test": {
            "format": "folder",
            "dataset_path": "data/vision/ChartQA/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Parquet shard file"
        },
        "train": {
            "format": "folder",
            "dataset_path": "data/vision/ChartQA/train",
            "ground_truth_path": None,
            "notes": "Train folder containing 3 Parquet shard files"
        },
        "val": {
            "format": "folder",
            "dataset_path": "data/vision/ChartQA/val",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Parquet shard file"
        },
    }


    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "ChartQA",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = "HuggingFaceM4/ChartQA",
        hf_repo_variant: str = "vis-nlp/ChartQA",
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING

    def load_split(
        self,
        dataset_split=None,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
    ):
        """Stream ChartQA samples in universal vision format. All splits have labels (test, train, val)."""
        if dataset_split or self.dataset_split:
            splits_to_load = [dataset_split or self.dataset_split]
        else:
            splits_to_load = list(self.FILE_MAPPING.keys())
        # ChartQA test/train/val all have ground truth; no filtering when only_splits_with_gt

        def _iter():
            emitted = 0
            for split in splits_to_load:
                raw_samples = self.vision_dataset_loader(
                    split,
                    self.FILE_MAPPING,
                    max_samples_per_split=max_samples_per_split,
                )

                for idx, row in enumerate(raw_samples):
                    yield {
                        "input": {
                            "image": row.get("image", row),
                            "question": row.get("query"),
                        },
                        # ChartQA ground truth labels are stored under "label" (list[str]).
                        # We keep the list here; evaluation will take the first element.
                        "ground_truth": row.get("label"),
                        "metadata": {
                            "dataset": "ChartQA",
                            "split": split,
                            "sample_id": f"{split}_{idx}",
                        },
                    }
                    emitted += 1
                    if max_samples_per_category and emitted >= max_samples_per_category:
                        return

        return _iter()


class MMMUAccountingAdapter(VisionDatasetAdapter):
    """
    data/vision/MMMU_Accounting/dev/data-00000-of-00001.parquet,
    is the only parquet shard file in dev/ sub-folder.
    
    === Dataset object ===
    Dataset({
        features: ['id', 'question', 'options', 'explanation', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'img_type', 'answer', 'topic_difficulty', 
    'question_type', 'subfield'],
        num_rows: 5
    })

    === Dataset features ===
    {'answer': Value('string'),
    'explanation': Value('string'),
    'id': Value('string'),
    'image_1': Image(mode=None, decode=True),
    'image_2': Image(mode=None, decode=True),
    'image_3': Image(mode=None, decode=True),
    'image_4': Image(mode=None, decode=True),
    'image_5': Image(mode=None, decode=True),
    'image_6': Image(mode=None, decode=True),
    'image_7': Image(mode=None, decode=True),
    'img_type': Value('string'),
    'options': Value('string'),
    'question': Value('string'),
    'question_type': Value('string'),
    'subfield': Value('string'),
    'topic_difficulty': Value('string')}

    === First 1 sample ===
    {'answer': 'D',
    'explanation': '',
    'id': 'dev_Accounting_1',
    'image_1': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1234x289 at 0x29A53A84FD0>,
    'image_2': None,
    'image_3': None,
    'image_4': None,
    'image_5': None,
    'image_6': None,
    'image_7': None,
    'img_type': "['Tables']",
    'options': "['$63,020', '$58,410', '$71,320', '$77,490']",
    'question': 'Each of the following situations relates to a different company. '
                '<image 1> For company B, find the missing amounts.',
    'question_type': 'multiple-choice',
    'subfield': 'Financial Accounting',
    'topic_difficulty': 'Easy'}


    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_mmmu_accounting_annotation().
    
    data/vision/MMMU_Accounting/test/data-00000-of-00001.parquet,
    is the only parquet shard file in test/ sub-folder.

    data/vision/MMMU_Accounting/validation/data-00000-of-00001.parquet,
    is the only parquet shard file in validation/ sub-folder.

    The sub-folder structure is consistent across dev, test, and validation splits:
    data/vision/MMMU_Accounting/dev/
    data/vision/MMMU_Accounting/test/
    data/vision/MMMU_Accounting/validation/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under dev/, test/, and validation/ folders.
    The parsing logic for each sample is implemented in the _parse_mmmu_accounting_annotation() function,
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    SPLITS_WITH_GT = {"dev", "test", "validation"}

    FILE_MAPPING = {
        "dev": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Accounting/dev",
            "ground_truth_path": None,
            "notes": "Dev folder containing 1 Parquet shard file"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Accounting/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Parquet shard file"
        },
        "validation": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Accounting/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Parquet shard file"
        },
    }
    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "MMMU_Accounting",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = "MMMU/MMMU",
        hf_repo_variant: str = "Accounting",
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING

    def load_split(
        self,
        dataset_split=None,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
    ):
        """Stream MMMU samples in universal vision format. Parquet rows have answer, question, image_1..image_7."""
        splits_to_load = (
            [dataset_split or self.dataset_split]
            if (dataset_split or self.dataset_split)
            else list(self.FILE_MAPPING.keys())
        )
        if only_splits_with_gt and getattr(self, "SPLITS_WITH_GT", None):
            splits_to_load = [s for s in splits_to_load if s in self.SPLITS_WITH_GT]

        def _iter():
            emitted = 0
            for split in splits_to_load:
                raw_samples = self.vision_dataset_loader(
                    split,
                    self.FILE_MAPPING,
                    max_samples_per_split=max_samples_per_split,
                )
                for idx, row in enumerate(raw_samples):
                    img = row.get("image_1")
                    if img is None:
                        for i in range(2, 8):
                            img = row.get(f"image_{i}")
                            if img is not None:
                                break
                    meta = {
                        "dataset": self.dataset_name,
                        "split": split,
                        "sample_id": row.get("id", f"{split}_{idx}"),
                    }
                    opts = _parse_options_list(row.get("options"))
                    if opts is not None:
                        meta["options_list"] = opts
                    yield {
                        "input": {"image": img, "question": row.get("question")},
                        "ground_truth": str(row.get("answer", "")),
                        "metadata": meta,
                    }
                    emitted += 1
                    if max_samples_per_category and emitted >= max_samples_per_category:
                        return
        return _iter()

class MMMUEconomicsAdapter(VisionDatasetAdapter):
    """
    data/vision/MMMU_Economics/dev/data-00000-of-00001.parquet,
    is the only parquet shard file in dev/ sub-folder.
    
    === Dataset object ===
    Dataset({
        features: ['id', 'question', 'options', 'explanation', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'img_type', 'answer', 'topic_difficulty', 
    'question_type', 'subfield'],
        num_rows: 5
    })

    === Dataset features ===
    {'answer': Value('string'),
    'explanation': Value('string'),
    'id': Value('string'),
    'image_1': Image(mode=None, decode=True),
    'image_2': Image(mode=None, decode=True),
    'image_3': Image(mode=None, decode=True),
    'image_4': Image(mode=None, decode=True),
    'image_5': Image(mode=None, decode=True),
    'image_6': Image(mode=None, decode=True),
    'image_7': Image(mode=None, decode=True),
    'img_type': Value('string'),
    'options': Value('string'),
    'question': Value('string'),
    'question_type': Value('string'),
    'subfield': Value('string'),
    'topic_difficulty': Value('string')}

    === First 1 samples ===
    {'answer': 'D',
    'explanation': '',
    'id': 'dev_Economics_1',
    'image_1': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=497x474 at 0x2310CFAABD0>,
    'image_2': None,
    'image_3': None,
    'image_4': None,
    'image_5': None,
    'image_6': None,
    'image_7': None,
    'img_type': "['Plots and Charts']",
    'options': "['500', '450', '400', '600']",
    'question': 'The graph below shows the AD-AS diagram for Spain. All numbers '
                'are in billions. <image 1> What is the price level in the '
                'short-run equilibrium?',
    'question_type': 'multiple-choice',
    'subfield': 'Macroeconomics',
    'topic_difficulty': 'Easy'}

 
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_mmmu_economics_annotation().
    
    data/vision/MMMU_Economics/test/data-00000-of-00001.parquet,
    is the only parquet shard file in test/ sub-folder.

    data/vision/MMMU_Economics/validation/data-00000-of-00001.parquet,
    is the only parquet shard file in validation/ sub-folder.

    The sub-folder structure is consistent across dev, test, and validation splits:
    data/vision/MMMU_Economics/dev/
    data/vision/MMMU_Economics/test/
    data/vision/MMMU_Economics/validation/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under dev/, test/, and validation/ folders.
    The parsing logic for each sample is implemented in the _parse_mmmu_economics_annotation() function,
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    SPLITS_WITH_GT = {"dev", "test", "validation"}

    FILE_MAPPING = {
        "dev": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Economics/dev",
            "ground_truth_path": None,
            "notes": "Dev folder containing 1 Parquet shard file"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Economics/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Parquet shard file"
        },
        "validation": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Economics/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Parquet shard file"
        },
    }
    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "MMMU_Economics",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = "MMMU/MMMU",
        hf_repo_variant: str = "Economics",
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING

    def load_split(
        self,
        dataset_split=None,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
    ):
        """Stream MMMU samples in universal vision format. Parquet rows have answer, question, image_1..image_7."""
        splits_to_load = (
            [dataset_split or self.dataset_split]
            if (dataset_split or self.dataset_split)
            else list(self.FILE_MAPPING.keys())
        )
        if only_splits_with_gt and getattr(self, "SPLITS_WITH_GT", None):
            splits_to_load = [s for s in splits_to_load if s in self.SPLITS_WITH_GT]

        def _iter():
            emitted = 0
            for split in splits_to_load:
                raw_samples = self.vision_dataset_loader(
                    split,
                    self.FILE_MAPPING,
                    max_samples_per_split=max_samples_per_split,
                )
                for idx, row in enumerate(raw_samples):
                    img = row.get("image_1")
                    if img is None:
                        for i in range(2, 8):
                            img = row.get(f"image_{i}")
                            if img is not None:
                                break
                    meta = {
                        "dataset": self.dataset_name,
                        "split": split,
                        "sample_id": row.get("id", f"{split}_{idx}"),
                    }
                    opts = _parse_options_list(row.get("options"))
                    if opts is not None:
                        meta["options_list"] = opts
                    yield {
                        "input": {"image": img, "question": row.get("question")},
                        "ground_truth": str(row.get("answer", "")),
                        "metadata": meta,
                    }
                    emitted += 1
                    if max_samples_per_category and emitted >= max_samples_per_category:
                        return
        return _iter()

class MMMUFinanceAdapter(VisionDatasetAdapter):
    """
    data/vision/MMMU_Finance/dev/data-00000-of-00001.parquet,
    is the only parquet shard file in dev/ sub-folder.
    
    === Dataset object ===
    Dataset({
        features: ['id', 'question', 'options', 'explanation', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'img_type', 'answer', 'topic_difficulty', 
    'question_type', 'subfield'],
        num_rows: 5
    })

    === Dataset features ===
    {'answer': Value('string'),
    'explanation': Value('string'),
    'id': Value('string'),
    'image_1': Image(mode=None, decode=True),
    'image_2': Image(mode=None, decode=True),
    'image_3': Image(mode=None, decode=True),
    'image_4': Image(mode=None, decode=True),
    'image_5': Image(mode=None, decode=True),
    'image_6': Image(mode=None, decode=True),
    'image_7': Image(mode=None, decode=True),
    'img_type': Value('string'),
    'options': Value('string'),
    'question': Value('string'),
    'question_type': Value('string'),
    'subfield': Value('string'),
    'topic_difficulty': Value('string')}

    === First 1 samples ===
    {'answer': '0.8638',
    'explanation': '',
    'id': 'dev_Finance_1',
    'image_1': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=577x219 at 0x1B7FA2A51D0>,
    'image_2': None,
    'image_3': None,
    'image_4': None,
    'image_5': None,
    'image_6': None,
    'image_7': None,
    'img_type': "['Tables']",
    'options': '[]',
    'question': 'Without referring to the preprogrammed function on your '
                'financial calculator, use the basic formula for present value, '
                'along with the given opportunity cost, r, and the number of '
                'periods, n, to calculate the present value of $1 in case C shown '
                'in the following table. <image 1>',
    'question_type': 'open',
    'subfield': 'Managerial Finance',
    'topic_difficulty': 'Easy'}

    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_mmmu_finance_annotation().
    
    data/vision/MMMU_Finance/test/data-00000-of-00001.parquet,
    is the only parquet shard file in test/ sub-folder.

    data/vision/MMMU_Finance/validation/data-00000-of-00001.parquet,
    is the only parquet shard file in validation/ sub-folder.

    The sub-folder structure is consistent across dev, test, and validation splits:
    data/vision/MMMU_Finance/dev/
    data/vision/MMMU_Finance/test/
    data/vision/MMMU_Finance/validation/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under dev/, test/, and validation/ folders.
    The parsing logic for each sample is implemented in the _parse_mmmu_finance_annotation() function,
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    SPLITS_WITH_GT = {"dev", "test", "validation"}

    FILE_MAPPING = {
        "dev": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Finance/dev",
            "ground_truth_path": None,
            "notes": "Dev folder containing 1 Parquet shard file"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Finance/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Parquet shard file"
        },
        "validation": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Finance/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Parquet shard file"
        },
    }
    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "MMMU_Finance",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = "MMMU/MMMU",
        hf_repo_variant: str = "Finance",
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING

    def load_split(
        self,
        dataset_split=None,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
    ):
        """Stream MMMU samples in universal vision format. Parquet rows have answer, question, image_1..image_7."""
        splits_to_load = (
            [dataset_split or self.dataset_split]
            if (dataset_split or self.dataset_split)
            else list(self.FILE_MAPPING.keys())
        )
        if only_splits_with_gt and getattr(self, "SPLITS_WITH_GT", None):
            splits_to_load = [s for s in splits_to_load if s in self.SPLITS_WITH_GT]

        def _iter():
            emitted = 0
            for split in splits_to_load:
                raw_samples = self.vision_dataset_loader(
                    split,
                    self.FILE_MAPPING,
                    max_samples_per_split=max_samples_per_split,
                )
                for idx, row in enumerate(raw_samples):
                    img = row.get("image_1")
                    if img is None:
                        for i in range(2, 8):
                            img = row.get(f"image_{i}")
                            if img is not None:
                                break
                    meta = {
                        "dataset": self.dataset_name,
                        "split": split,
                        "sample_id": row.get("id", f"{split}_{idx}"),
                    }
                    opts = _parse_options_list(row.get("options"))
                    if opts is not None:
                        meta["options_list"] = opts
                    yield {
                        "input": {"image": img, "question": row.get("question")},
                        "ground_truth": str(row.get("answer", "")),
                        "metadata": meta,
                    }
                    emitted += 1
                    if max_samples_per_category and emitted >= max_samples_per_category:
                        return
        return _iter()

class MMMUMathAdapter(VisionDatasetAdapter):
    """
    data/vision/MMMU_Math/dev/data-00000-of-00001.parquet,
    is the only parquet shard file in dev/ sub-folder.
    
    === Dataset object ===
    Dataset({
        features: ['id', 'question', 'options', 'explanation', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'img_type', 'answer', 'topic_difficulty', 
    'question_type', 'subfield'],
        num_rows: 5
    })

    === Dataset features ===
    {'answer': Value('string'),
    'explanation': Value('string'),
    'id': Value('string'),
    'image_1': Image(mode=None, decode=True),
    'image_2': Image(mode=None, decode=True),
    'image_3': Image(mode=None, decode=True),
    'image_4': Image(mode=None, decode=True),
    'image_5': Image(mode=None, decode=True),
    'image_6': Image(mode=None, decode=True),
    'image_7': Image(mode=None, decode=True),
    'img_type': Value('string'),
    'options': Value('string'),
    'question': Value('string'),
    'question_type': Value('string'),
    'subfield': Value('string'),
    'topic_difficulty': Value('string')}

    === First 1 samples ===
    {'answer': '4',
    'explanation': '',
    'id': 'dev_Math_1',
    'image_1': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=744x261 at 0x2A41F365010>,
    'image_2': None,
    'image_3': None,
    'image_4': None,
    'image_5': None,
    'image_6': None,
    'image_7': None,
    'img_type': "['Tables']",
    'options': '[]',
    'question': 'Each of seven students has chosen three courses from ten '
                'options, and must sit an exam for each of his or her three '
                'choices. Two students sitting the same exam must do so at the '
                'same time, but no student can sit more than one exam in the same '
                'day. The table of choices is given in <image 1>. Find the '
                'smallest number of days required to schedule the exams. Return '
                'only the number of days.',
    'question_type': 'open',
    'subfield': 'Graph Theory',
    'topic_difficulty': 'Easy'}

    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_mmmu_math_annotation().
    
    data/vision/MMMU_Math/test/data-00000-of-00001.parquet,
    is the only parquet shard file in test/ sub-folder.

    data/vision/MMMU_Math/validation/data-00000-of-00001.parquet,
    is the only parquet shard file in validation/ sub-folder.

    The sub-folder structure is consistent across dev, test, and validation splits:
    data/vision/MMMU_Math/dev/
    data/vision/MMMU_Math/test/
    data/vision/MMMU_Math/validation/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under dev/, test/, and validation/ folders.
    The parsing logic for each sample is implemented in the _parse_mmmu_math_annotation() function,
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    SPLITS_WITH_GT = {"dev", "test", "validation"}

    FILE_MAPPING = {
        "dev": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Math/dev",
            "ground_truth_path": None,
            "notes": "Dev folder containing 1 Parquet shard file"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Math/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Parquet shard file"
        },
        "validation": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Math/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Parquet shard file"
        },
    }
    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "MMMU_Math",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = "MMMU/MMMU",
        hf_repo_variant: str = "Math",
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING

    def load_split(
        self,
        dataset_split=None,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
    ):
        """Stream MMMU samples in universal vision format. Parquet rows have answer, question, image_1..image_7."""
        splits_to_load = (
            [dataset_split or self.dataset_split]
            if (dataset_split or self.dataset_split)
            else list(self.FILE_MAPPING.keys())
        )
        if only_splits_with_gt and getattr(self, "SPLITS_WITH_GT", None):
            splits_to_load = [s for s in splits_to_load if s in self.SPLITS_WITH_GT]

        def _iter():
            emitted = 0
            for split in splits_to_load:
                raw_samples = self.vision_dataset_loader(
                    split,
                    self.FILE_MAPPING,
                    max_samples_per_split=max_samples_per_split,
                )
                for idx, row in enumerate(raw_samples):
                    img = row.get("image_1")
                    if img is None:
                        for i in range(2, 8):
                            img = row.get(f"image_{i}")
                            if img is not None:
                                break
                    meta = {
                        "dataset": self.dataset_name,
                        "split": split,
                        "sample_id": row.get("id", f"{split}_{idx}"),
                    }
                    opts = _parse_options_list(row.get("options"))
                    if opts is not None:
                        meta["options_list"] = opts
                    yield {
                        "input": {"image": img, "question": row.get("question")},
                        "ground_truth": str(row.get("answer", "")),
                        "metadata": meta,
                    }
                    emitted += 1
                    if max_samples_per_category and emitted >= max_samples_per_category:
                        return
        return _iter()

# ------------------------
# RAG Adapters
# ------------------------
"""
universal_rag_sample = {
    "input": {
        "query": str,
        "context": Any,   # optional table/paragraph/chunk payload
        "image": <PIL.Image.Image> or None,
    },
    "ground_truth": {
        "answer": str,
        "scale": str | None,
        "extra": dict | None,
    },
    "metadata": {
        "dataset": str,
        "split": str,
        "sample_id": str | int,
    }
}
"""
class FinQAAdapter(BaseDatasetAdapter):
    """
    FinQA for RAG QA evaluation. Data is loaded from a single JSON file per split.

    data/rag/FinQA/train/train_qa.json

    === Dataset schema (train_qa.json) ===
    List of entries; each entry has:
    - "id": str (unique example id, e.g. report name + index)
    - "pre_text": str (text before the table)
    - "post_text": str (text after the table)
    - "table": list (structured table data)
    - "qa": {
        "question": str,
        "program": list (reasoning program),
        "gold_inds": ...,
        "exe_ans": str (gold execution result / answer),
        "program_re": ...
      }

    === First sample (structure) ===
    {
        "id": "...",
        "pre_text": "...",
        "post_text": "...",
        "table": [...],
        "qa": {
            "question": "What is ...?",
            "exe_ans": "123.45",
            "program": [...]
        }
    }

    FILE_MAPPING points to the JSON file path. No parquet is used.
    Source: Official FinQA from https://github.com/czyssrs/FinQA (dataset/train.json).

    Only the train split is configured. Official FinQA has train/dev/test; we evaluate on train
    only (index is built from train, and we do not load a test split — test-set ground truth
    may be public or held-out depending on source; this codebase uses train for both indexing and eval).
    """
    FILE_MAPPING = {
        "train": {
            "format": "json",
            "dataset_path": "data/rag/FinQA/train/train_qa.json",
            "ground_truth_path": None,
            "notes": "Single JSON file train_qa.json (list of QA entries)"
        }
    }
    SPLITS_WITH_GT = {"train"}

    def __init__(
        self,
        category: str = "rag",
        dataset_name: str = "FinQA",
        data_source_from_hf_or_manual: str = "manual",
        hf_repo_name: str = "FinanceMTEB/FinQA",
        hf_repo_variant: str = None,
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING

    def load_split(
        self,
        dataset_split=None,
        recursive=False,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
    ):
        """Stream FinQA split(s) in universal RAG format from train_qa.json. All splits have ground truth."""
        splits_to_load = (
            [dataset_split or self.dataset_split]
            if (dataset_split or self.dataset_split)
            else list(self.FILE_MAPPING.keys())
        )
        if only_splits_with_gt and getattr(self, "SPLITS_WITH_GT", None):
            splits_to_load = [s for s in splits_to_load if s in self.SPLITS_WITH_GT]

        def _load_qa_list(path: Path) -> list:
            """Load list of QA entries from JSON (list or dict with 'data' key)."""
            if not path.exists():
                return []
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return []

        def _entry_to_query_and_answer(entry: dict) -> tuple:
            """Extract query and ground-truth answer from one train_qa.json entry."""
            qa = entry.get("qa") or {}
            if isinstance(qa, dict):
                query = qa.get("question") or entry.get("question") or entry.get("query")
                ans = qa.get("exe_ans") or qa.get("answer") or entry.get("answer") or entry.get("answers")
            else:
                query = entry.get("question") or entry.get("query") or (entry.get("questions", [None])[0] if entry.get("questions") else None)
                ans = entry.get("answer") or entry.get("answers")
            gt_answer = ans[0] if isinstance(ans, list) else (str(ans) if ans is not None else "")
            return query or "", gt_answer or ""

        def _iter():
            emitted = 0
            for split in splits_to_load:
                if split not in self.FILE_MAPPING:
                    raise ValueError(f"FinQA supports splits: {list(self.FILE_MAPPING.keys())}")

                json_path = Path().cwd() / self.FILE_MAPPING[split]["dataset_path"]
                qa_list = _load_qa_list(json_path)
                if not qa_list:
                    raise FileNotFoundError(
                        f"FinQA {split}: {json_path} not found or empty. "
                        "Add train_qa.json from https://github.com/czyssrs/FinQA (dataset/train.json)."
                    )

                for idx, entry in enumerate(qa_list):
                    if max_samples_per_split is not None and idx >= max_samples_per_split:
                        break
                    query, gt_answer = _entry_to_query_and_answer(entry)
                    qa = entry.get("qa") or {}
                    gold_program = qa.get("program")
                    if isinstance(gold_program, list):
                        gold_program = str(gold_program) if gold_program else None
                    yield {
                        "input": {
                            "query": query,
                            "image": None,
                        },
                        "ground_truth": {
                            "answer": gt_answer,
                            "score": None,
                            "query_id": idx,
                            "corpus_id": entry.get("id") or entry.get("filename"),
                            **({"program": gold_program} if gold_program is not None else {}),
                        },
                        "metadata": {
                            "dataset": self.dataset_name,
                            "split": split,
                            "sample_id": entry.get("id", idx),
                        },
                    }
                    emitted += 1
                    if max_samples_per_category is not None and emitted >= max_samples_per_category:
                        return

        return _iter()

class TATQAAdapter(BaseDatasetAdapter):
    """
    data/rag/TAT-QA/tatqa_dataset_dev.json
    
    [
        {
            "table": {
            "uid": "3ffd9053-a45d-491c-957a-1b2fa0af0570",
            "table": [
                [
                "",
                "",
                "Years Ended September 30,",
                ""
                ],
                [
                "",
                "2019",
                "2018",
                "2017"
                ],
                [
                "Fixed Price",
                "$  1,452.4",
                "$  1,146.2",
                "$  1,036.9"
                ],
                [
                "Other",
                "44.1",
                "56.7",
                "70.8"
                ],
                [
                "Total sales",
                "$1,496.5",
                "$1,202.9",
                "$1,107.7"
                ]
            ]
            },
            "paragraphs": [
            {
                "uid": "f4ac7069-10a2-47e9-995c-3903293b3d47",
                "order": 1,
                "text": "Sales by Contract Type: Substantially all of our contracts are fixed-price type contracts. Sales included in Other contract types represent cost plus and time and material type contracts."
            },
            {
                "uid": "79e37805-6558-4a8c-b033-32be6bffef48",
                "order": 2,
                "text": "On a fixed-price type contract, we agree to perform the contractual statement of work for a predetermined sales price. On a cost-plus type contract, we are paid our allowable incurred costs plus a profit which can be fixed or variable depending on the contract’s fee arrangement up to predetermined funding levels determined by the customer. On a time-and-material type contract, we are paid on the basis of direct labor hours expended at specified fixed-price hourly rates (that include wages, overhead, allowable general and administrative expenses and profit) and materials at cost. The table below presents total net sales disaggregated by contract type (in millions):"
            }
            ],
            "questions": [
            {
                "uid": "23801627-ff77-4597-8d24-1c99e2452082",
                "order": 1,
                "question": "What is the company paid on a cost-plus type contract?",
                "answer": [
                "our allowable incurred costs plus a profit which can be fixed or variable depending on the contract’s fee arrangement up to predetermined funding levels determined by the customer"
                ],
                "derivation": "",
                "answer_type": "span",
                "answer_from": "text",
                "rel_paragraphs": [
                "2"
                ],
                "req_comparison": false,
                "scale": ""
            },
            {
                "uid": "4960801d-277d-4f79-8eca-c4d0200fa9d6",
                "order": 2,
                "question": "What is the amount of total sales in 2019?",
                "answer": [
                "$1,496.5"
                ],
                "derivation": "",
                "answer_type": "span",
                "answer_from": "table-text",
                "rel_paragraphs": [
                "2"
                ],
                "req_comparison": false,
                "scale": "million"
            },
            {
                "uid": "593c4388-5209-4462-8b83-b429c8612c25",
                "order": 3,
                "question": "What are the contract types?",
                "answer": [
                "fixed-price type",
                "cost-plus type",
                "time-and-material type"
                ],
                "derivation": "",
                "answer_type": "multi-span",
                "answer_from": "text",
                "rel_paragraphs": [
                "1",
                "2"
                ],
                "req_comparison": false,
                "scale": ""
            },
            {
                "uid": "f4142349-eb72-49eb-9a76-f3ccb1010cbc",
                "order": 4,
                "question": "In which year is the amount of total sales the largest?",
                "answer": [
                "2019"
                ],
                "derivation": "1,496.5>1,202.9>1,107.7",
                "answer_type": "span",
                "answer_from": "table-text",
                "rel_paragraphs": [
                "2"
                ],
                "req_comparison": true,
                "scale": ""
            },
            {
                "uid": "eb787966-fa02-401f-bfaf-ccabf3828b23",
                "order": 5,
                "question": "What is the change in Other in 2019 from 2018?",
                "answer": -12.6,
                "derivation": "44.1-56.7",
                "answer_type": "arithmetic",
                "answer_from": "table-text",
                "rel_paragraphs": [
                "2"
                ],
                "req_comparison": false,
                "scale": "million"
            },
            {
                "uid": "05b670d3-5b19-438c-873f-9bf6de29c69e",
                "order": 6,
                "question": "What is the percentage change in Other in 2019 from 2018?",
                "answer": -22.22,
                "derivation": "(44.1-56.7)/56.7",
                "answer_type": "arithmetic",
                "answer_from": "table-text",
                "rel_paragraphs": [
                "2"
                ],
                "req_comparison": false,
                "scale": "percent"
            }
            ]
        },
        {
    
    data/rag/TAT-QA/tatqa_dataset_test_gold.json,
    data/rag/TAT-QA/tatqa_dataset_test.json,
    data/rag/TAT-QA/tatqa_dataset_train.json.
    There are only these 4 json files (dev, test, ground truth for test, train) under TAT-QA/ folder,
    and they all follow the same schema structure as above, but with different samples.

    We evaluate on test only (FILE_MAPPING and SPLITS_WITH_GT). The RAG retriever index must be built
    from test documents (tatqa_dataset_test_gold.json) so retrieval can find the right context for
    each test question. Train and dev are not used for evaluation in this codebase.

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under TAT-QA/ folder.
    as well as as the pathing down to the individual sample level.
    The parsing logic for each sample is implemented in the _parse_tatqa_sample() function, 
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    FILE_MAPPING = {
        # "dev": {  # no official GT file for eval; evaluation uses test (tatqa_dataset_test_gold.json) only
        #     "format": "json",
        #     "dataset_path": "data/rag/TAT-QA/dev.json",
        #     "ground_truth_path": None,
        #     "notes": None
        # },
        "test": {
            "format": "json",
            "dataset_path": "data/rag/TAT-QA/test.json",
            "ground_truth_path": "data/rag/TAT-QA/test_gold.json",
            "notes": "2 json files: Ground truth for test set is in 'test_gold.json', test set is in 'test.json'"
        },
        # "train": {  # no official GT file for eval; evaluation uses test only
        #     "format": "json",
        #     "dataset_path": "data/rag/TAT-QA/train.json",
        #     "ground_truth_path": None,
        #     "notes": None
        # },
    }
    SPLITS_WITH_GT = {"test"}

    def __init__(
        self,
        category: str = "rag",
        dataset_name: str = "TATQA",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = None, # "next-tat/TAT-QA"
        hf_repo_variant: str = None,
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING
        # Data lives under TAT-QA folder; adapter name is TATQA (no hyphen)
        self.root_dir = os.path.join(os.getcwd(), "data", self.category, "TAT-QA")

    def load_split(
        self,
        dataset_split=None,
        recursive=False,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
    ):
        """Stream TAT-QA split(s) in universal RAG format. All splits have ground truth."""
        split_files = {
            "train": "tatqa_dataset_train.json",
            "dev": "tatqa_dataset_dev.json",
            "test": "tatqa_dataset_test_gold.json",
        }
        splits_to_load = (
            [dataset_split or self.dataset_split]
            if (dataset_split or self.dataset_split)
            else list(self.FILE_MAPPING.keys())
        )
        if only_splits_with_gt and getattr(self, "SPLITS_WITH_GT", None):
            splits_to_load = [s for s in splits_to_load if s in self.SPLITS_WITH_GT]

        def _iter():
            emitted = 0
            for split in splits_to_load:
                file_name = split_files.get(split)
                if not file_name:
                    continue
                path = Path(self.root_dir) / file_name
                if not path.exists():
                    print(f"⚠️ {self.dataset_name} {split} split not found at {path}")
                    continue

                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                sample_idx = 0
                for doc in data:
                    table = doc.get("table", {})
                    paragraphs = doc.get("paragraphs", [])
                    for q in doc.get("questions", []):
                        if max_samples_per_split is not None and sample_idx >= max_samples_per_split:
                            break
                        answers = q.get("answer", [])
                        if isinstance(answers, list):
                            answer = answers[0] if answers else ""
                        else:
                            answer = answers

                        yield {
                            "input": {
                                "query": q.get("question"),
                                "context": {"table": table.get("table"), "paragraphs": paragraphs},
                                "image": None,
                            },
                            "ground_truth": {
                                "answer": str(answer),
                                "scale": q.get("scale", ""),
                                "derivation": q.get("derivation", ""),
                            },
                            "metadata": {
                                "dataset": self.dataset_name,
                                "split": split,
                                "sample_id": q.get("uid", f"{split}_{sample_idx}"),
                            },
                        }
                        sample_idx += 1
                        emitted += 1
                        if max_samples_per_category and emitted >= max_samples_per_category:
                            return

        return _iter()

# ------------------------
# Credit Risk Adapters (PD)
# ------------------------
"""
universal_credit_risk_pd_sample = {
    "input": {
        "features": dict | str,
    },
    "ground_truth": {
        "label": str | int | float,
    },
    "metadata": {
        "dataset": str,
        "split": str,
        "sample_id": str | int,
    }
}
"""
class LendingClubAdapter(BaseDatasetAdapter):
    """
    data/credit_risk_pd/LendingClub/test/data-00000-of-00001.parquet,
    is the only parquet shard file in test/ sub-folder.
    
    === Dataset object ===
    Dataset({
        features: ['id', 'query', 'answer', 'choices', 'gold'],
        num_rows: 2691
    })

    === Dataset features ===
    {'answer': Value('string'),
    'choices': List(Value('string')),
    'gold': Value('int64'),
    'id': Value('string'),
    'query': Value('string')}

    === First 1 sample ===
    {'answer': 'fullypaid',
    'choices': ['fullypaid', 'chargedoff'],
    'gold': 0,
    'id': 'ex_000001',
    'query': 'Predict the loan status using the features below. Directly respond '
            "with 'fullypaid' if the loan is fully repaid or 'chargeoff' if the "
            'loan is charged off.\n'
            "Text: ' addressState: ga, annualIncome: 66400.00, "
            'delinquencyIn2Years: 2.00, employmentLength: 10+Years, '
            'ficoRangeHigh: 669.00, ficoRangeLow: 665.00, grade: f, '
            'homeOwnership: mortgage, inquiriesIn6Months: 1.00, installment: '
            '267.07, interestRate: 24.08, lastPaymentAmount: 267.02, loanAmount: '
            '6800.00, loanApplicationType: individual, loanPurpose: other, '
            'mortgageAccounts: 4.00, openAccounts: 14.00, revolvingBalance: '
            '17607.00, revolvingUtilizationRate: 91.70, totalAccounts: 22.00, '
            "verificationStatus: notVerified '.\n"
            'Answer:'}

    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_lendingclub_annotation().
    
    data/credit_risk_pd/LendingClub/train/data-00000-of-00001.parquet,
    is the only parquet shard file in train/ sub-folder.

    data/credit_risk_pd/LendingClub/valid/data-00000-of-00001.parquet,
    is the only parquet shard file in valid/ sub-folder.

    The sub-folder structure is consistent across test, train, and valid splits:
    data/credit_risk_pd/LendingClub/test/
    data/credit_risk_pd/LendingClub/train/
    data/credit_risk_pd/LendingClub/valid/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under test/, train/, and valid/ folders.
    The parsing logic for each sample is implemented in the _parse_lendingclub_annotation() function,
    which is called by the generic _load_samples() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    FILE_MAPPING = {
        "test": {
            "format": "folder",
            "dataset_path": "data/credit_risk_pd/LendingClub/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Parquet shard file"
        },
        "train": {
            "format": "folder",
            "dataset_path": "data/credit_risk_pd/LendingClub/train",
            "ground_truth_path": None,
            "notes": "Train folder (for full overnight sample-by-sample evaluation)"
        },
        "valid": {
            "format": "folder",
            "dataset_path": "data/credit_risk_pd/LendingClub/valid",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Parquet shard file"
        },
    }
    def __init__(
        self,
        category: str = "credit_risk_pd",
        dataset_name: str = "LendingClub",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = "TheFinAI/lendingclub-benchmark",
        hf_repo_variant: str = None,
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING

    def load_split(
        self,
        dataset_split=None,
        recursive=False,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
        **kwargs,
    ):
        """Stream samples from parquet (test/valid per FILE_MAPPING). One row at a time; parse query text to features.
        When max_samples_per_split is set, samples are drawn in a stratified way (both gold=0 and gold=1) so that
        AUC-ROC is defined (otherwise the first N rows might be single-class and auc_roc_mean would be 0.5).
        """
        from credit_risk.feature_engineering.common_features import parse_query_to_features
        import random

        def _row_to_sample(row, split: str, idx: int):
            query = row.get("query") or ""
            gold = row.get("gold")
            answer = row.get("answer")
            if gold is not None and hasattr(gold, "item"):
                gold = int(gold.item())
            elif answer is not None:
                gold = 1 if str(answer).strip().lower() in ("chargedoff", "chargeoff") else 0
            else:
                gold = None
            features = parse_query_to_features(query, use_no_leakage=True)
            return {
                "input": {"features": features, "query": query},
                "ground_truth": {"label": gold} if gold is not None else {},
                "metadata": {
                    "dataset": self.dataset_name,
                    "split": split,
                    "sample_id": row.get("id") or f"{split}_{idx}",
                },
            }

        splits_to_load = (
            [dataset_split] if dataset_split is not None
            else list(self.FILE_MAPPING.keys())
        )
        emitted = 0
        for split in splits_to_load:
            if split not in self.FILE_MAPPING:
                continue
            split_path = Path().cwd() / self.FILE_MAPPING[split]["dataset_path"]
            if not split_path.exists():
                continue

            if max_samples_per_split is not None and max_samples_per_split > 0:
                # Stratified sampling: collect both classes so AUC is defined
                cap = min(2 * max_samples_per_split + 5000, 100_000)
                neg, pos = [], []
                for idx, row in enumerate(self._arrow_row_stream(split_path, max_samples=cap)):
                    gold = row.get("gold")
                    if gold is not None and hasattr(gold, "item"):
                        gold = int(gold.item())
                    elif row.get("answer") is not None:
                        gold = 1 if str(row.get("answer", "")).strip().lower() in ("chargedoff", "chargeoff") else 0
                    else:
                        gold = None
                    if gold is None:
                        continue
                    sample = _row_to_sample(row, split, idx)
                    if gold == 1:
                        pos.append(sample)
                    else:
                        neg.append(sample)
                    if len(pos) + len(neg) >= max_samples_per_split * 2:
                        break
                # Yield up to max_samples_per_split, balancing both classes
                n_neg = min(len(neg), (max_samples_per_split + 1) // 2)
                n_pos = min(len(pos), max_samples_per_split - n_neg)
                if n_neg + n_pos < max_samples_per_split and (len(neg) + len(pos)) >= max_samples_per_split:
                    n_neg = min(len(neg), max_samples_per_split - n_pos)
                rng = random.Random(42)
                neg_sub = rng.sample(neg, n_neg) if len(neg) > n_neg else neg
                pos_sub = rng.sample(pos, n_pos) if len(pos) > n_pos else pos
                for s in (neg_sub + pos_sub)[:max_samples_per_split]:
                    yield s
                    emitted += 1
                    if max_samples_per_category is not None and emitted >= max_samples_per_category:
                        return
                continue

            for idx, row in enumerate(self._arrow_row_stream(split_path, max_samples=max_samples_per_split)):
                yield _row_to_sample(row, split, idx)
                emitted += 1
                if max_samples_per_category is not None and emitted >= max_samples_per_category:
                    return

# ------------------------
# Credit Risk Adapters (Sentiment)
# ------------------------

"""
universal_credit_risk_sentiment_sample = {
    "input": {"text": str},
    "ground_truth": {"label": str | int},
    "metadata": {"dataset": str, "split": str, "sample_id": str | int}
}
"""

class FinancialPhraseBankAdapter(BaseDatasetAdapter):
    """
    data/credit_risk_sentiment/FinancialPhraseBank/test/data-00000-of-00001.parquet,
    is the only parquet shard file in test/ sub-folder.
    
    === Dataset features ===
    {'label': Value('int64'),
    'label_text': Value('string'),
    'text': Value('string')}

    === First 2 samples ===
    {'label': 2,
    'label_text': 'positive',
    'text': 'Rautaruukki said construction group YIT has awarded it a 2.5 mln eur '
            'contract to supply the steel structures for a new bridge spanning '
            'the Kemijoki river in Northern Finland .'}

    --- Sample 1 ---
    {'label': 1,
    'label_text': 'neutral',
    'text': 'Finnish Raute Precision that supplies weighing and dosing systems '
            'and plants is changing its name to Lahti Precision .'}
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_financial_phrase_bank_annotation().
    
    data/credit_risk_sentiment/FinancialPhraseBank/train/data-00000-of-00001.parquet,
    is the only parquet shard file in train/ sub-folder.

    The sub-folder structure is consistent across test and train splits:
    data/credit_risk_sentiment/FinancialPhraseBank/test/
    data/credit_risk_sentiment/FinancialPhraseBank/train/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under test/ and train/ folders.
    The parsing logic for each sample is implemented in the _parse_financial_phrase_bank_annotation() function,
    which is called by the generic _load_samples() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    FILE_MAPPING = {
        "test": {
            "format": "folder",
            "dataset_path": "data/credit_risk_sentiment/FinancialPhraseBank/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Parquet shard file"
        },
        # "train": {  # for evaluation only splits with GT used for reporting; train used for training only
        #     "format": "folder",
        #     "dataset_path": "data/credit_risk_sentiment/FinancialPhraseBank/train",
        #     "ground_truth_path": None,
        #     "notes": "Train folder containing 1 Parquet shard file"
        # },
    }
    def __init__(
        self,
        category: str = "credit_risk_sentiment",
        dataset_name: str = "FinancialPhraseBank",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = "takala/financial_phrasebank",
        hf_repo_variant: str = "sentences_allagree",
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING

    def load_split(
        self,
        dataset_split=None,
        recursive=False,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
        **kwargs,
    ):
        """Stream samples from FILE_MAPPING (test only for eval). Parquet rows: label, label_text, text."""
        splits_to_load = (
            [dataset_split or self.dataset_split]
            if (dataset_split or self.dataset_split)
            else list(self.FILE_MAPPING.keys())
        )
        emitted = 0
        for split in splits_to_load:
            if split not in self.FILE_MAPPING:
                continue
            split_path = Path().cwd() / self.FILE_MAPPING[split]["dataset_path"]
            for idx, row in enumerate(self._arrow_row_stream(split_path, max_samples=max_samples_per_split)):
                label = row.get("label")
                if label is not None and hasattr(label, "item"):
                    label = label.item()
                yield {
                    "input": {"text": row.get("text") or row.get("sentence", "")},
                    "ground_truth": row.get("label_text") or str(label) if label is not None else "",
                    "metadata": {
                        "dataset": self.dataset_name,
                        "split": split,
                        "sample_id": row.get("id", f"{split}_{idx}"),
                    },
                }
                emitted += 1
                if max_samples_per_category and emitted >= max_samples_per_category:
                    return

class FiQAAdapter(BaseDatasetAdapter):
    """
    data/credit_risk_sentiment/FiQA/test/data-00000-of-00001.parquet,
    is the only parquet shard file in test/ sub-folder.
    
    === Dataset object ===
    Dataset({
        features: ['_id', 'sentence', 'target', 'aspect', 'score', 'type'],
        num_rows: 234
    })

    === Dataset features ===
    {'_id': Value('string'),
    'aspect': Value('string'),
    'score': Value('float64'),
    'sentence': Value('string'),
    'target': Value('string'),
    'type': Value('string')}

    === First 1 sample ===
    {'_id': '645',
    'aspect': 'Stock/Price Action',
    'score': 0.329,
    'sentence': "Britain's FTSE steadies, supported by Dixons Carphone",
    'target': 'Dixons Carphone',
    'type': 'headline'}

    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_fiqa_annotation().
    
    data/credit_risk_sentiment/FiQA/train/data-00000-of-00001.parquet,
    is the only parquet shard file in train/ sub-folder.

    data/credit_risk_sentiment/FiQA/valid/data-00000-of-00001.parquet,
    is the only parquet shard file in valid/ sub-folder.

    The sub-folder structure is consistent across test, train, and valid splits:
    data/credit_risk_sentiment/FiQA/test/
    data/credit_risk_sentiment/FiQA/train/
    data/credit_risk_sentiment/FiQA/valid/

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under test/, train/, and valid/ folders.
    The parsing logic for each sample is implemented in the _parse_fiqa_annotation() function,
    which is called by the generic _load_samples() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    FILE_MAPPING = {
        "test": {
            "format": "folder",
            "dataset_path": "data/credit_risk_sentiment/FiQA/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Parquet shard file"
        },
        "train": {
            "format": "folder",
            "dataset_path": "data/credit_risk_sentiment/FiQA/train",
            "ground_truth_path": None,
            "notes": "Train folder for training; eval uses test/valid"
        },
        "valid": {
            "format": "folder",
            "dataset_path": "data/credit_risk_sentiment/FiQA/valid",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Parquet shard file"
        },
    }

    def __init__(
        self,
        category: str = "credit_risk_sentiment",
        dataset_name: str = "FiQA",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = "TheFinAI/fiqa-sentiment-classification",
        hf_repo_variant: str = None,
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING

    def load_split(
        self,
        dataset_split=None,
        recursive=False,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
        **kwargs,
    ):
        """Stream samples from parquet (test/train/valid per FILE_MAPPING). Score mapped to label (negative/neutral/positive)."""
        def _score_to_label(score: float, neg: float = -0.2, pos: float = 0.2) -> str:
            if score < neg:
                return "negative"
            if score > pos:
                return "positive"
            return "neutral"
        splits_to_load = (
            [dataset_split] if dataset_split is not None
            else list(self.FILE_MAPPING.keys())
        )
        emitted = 0
        for split in splits_to_load:
            if split not in self.FILE_MAPPING:
                continue
            split_path = Path().cwd() / self.FILE_MAPPING[split]["dataset_path"]
            if not split_path.exists():
                continue
            for idx, row in enumerate(self._arrow_row_stream(split_path, max_samples=max_samples_per_split)):
                text = row.get("sentence") or row.get("text") or ""
                score = row.get("score")
                if score is not None and hasattr(score, "item"):
                    score = float(score.item())
                elif score is not None:
                    score = float(score)
                else:
                    score = 0.0
                label_text = _score_to_label(score)
                yield {
                    "input": {"text": text},
                    "ground_truth": label_text,
                    "metadata": {
                        "dataset": self.dataset_name,
                        "split": split,
                        "sample_id": row.get("_id") or row.get("id") or f"{split}_{idx}",
                    },
                }
                emitted += 1
                if max_samples_per_category is not None and emitted >= max_samples_per_category:
                    return

# ------------------------
# Credit Risk Adapters (Memo Generator)
# ------------------------

"""
universal_credit_risk_memo_generator_sample = {
    "input": {"prompt": str, "context": Any},
    "ground_truth": {"reference": str | None},
    "metadata": {"dataset": str, "split": str, "sample_id": str | int}
}
"""

class FinanceBenchAdapter(BaseDatasetAdapter):
    """
    data/credit_risk_memo_generator/FinanceBench/train/data-00000-of-00001.parquet,
    is the only parquet shard file in train/ sub-folder.
    
    === Dataset object ===
    Dataset({
        features: ['financebench_id', 'company', 'doc_name', 'question_type', 'question_reasoning', 'domain_question_num', 'question', 'answer', 'justification', 'dataset_subset_label', 'evidence', 'gics_sector', 'doc_type', 'doc_period', 'doc_link'],
        num_rows: 150
    })

    === Dataset features ===
    {'answer': Value('string'),
    'company': Value('string'),
    'dataset_subset_label': Value('string'),
    'doc_link': Value('string'),
    'doc_name': Value('string'),
    'doc_period': Value('int64'),
    'doc_type': Value('string'),
    'domain_question_num': Value('string'),
    'evidence': List({'evidence_text': Value('string'), 'doc_name': Value('string'), 'evidence_page_num': Value('int64'), 'evidence_text_full_page': Value('string')}),
    'financebench_id': Value('string'),
    'gics_sector': Value('string'),
    'justification': Value('string'),
    'question': Value('string'),
    'question_reasoning': Value('string'),
    'question_type': Value('string')}
    
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_financebench_annotation().

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under train/ folder.
    The parsing logic for each sample is implemented in the _parse_financebench_annotation() function,
    which is called by the generic _load_samples() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    FILE_MAPPING = {
        "default": {
            "format": "folder",
            "dataset_path": "data/credit_risk_memo_generator/FinanceBench/train",
            "ground_truth_path": None,
            "notes": "Single parquet file under 'train'; no official splits"
        }
    }
    def __init__(
        self,
        category: str = "credit_risk_memo_generator",
        dataset_name: str = "FinanceBench",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = "PatronusAI/financebench",
        hf_repo_variant: str = None,
        dataset_split=None
    ):
        super().__init__(
            category=category,
            dataset_name=dataset_name,
            data_source_from_hf_or_manual=data_source_from_hf_or_manual,
            hf_repo_name=hf_repo_name,
            hf_repo_variant=hf_repo_variant
        )
        self.category = category
        self.dataset_name = dataset_name
        self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
        self.hf_repo_name = hf_repo_name
        self.hf_repo_variant = hf_repo_variant
        self.dataset_split = dataset_split
        self.FILE_MAPPING = self.FILE_MAPPING

    def load_split(
        self,
        dataset_split=None,
        recursive=False,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=False,
        **kwargs,
    ):
        """Stream samples from parquet (train per FILE_MAPPING). Context = concatenated evidence text."""
        split_name = dataset_split or self.dataset_split or "default"
        splits_to_load = [split_name] if split_name in self.FILE_MAPPING else list(self.FILE_MAPPING.keys())
        emitted = 0
        for split in splits_to_load:
            if split not in self.FILE_MAPPING:
                continue
            split_path = Path().cwd() / self.FILE_MAPPING[split]["dataset_path"]
            if not split_path.exists():
                continue
            for idx, row in enumerate(self._arrow_row_stream(split_path, max_samples=max_samples_per_split)):
                question = row.get("question") or ""
                evidence = row.get("evidence") or []
                context_parts = []
                if isinstance(evidence, list):
                    for e in evidence:
                        if isinstance(e, dict) and e.get("evidence_text"):
                            context_parts.append(e["evidence_text"])
                        else:
                            context_parts.append(str(e))
                else:
                    context_parts.append(str(evidence))
                context = "\n\n".join(context_parts)
                answer = row.get("answer")
                yield {
                    "input": {"question": question, "context": context, "prompt": question},
                    "ground_truth": {"reference": answer} if answer is not None else {},
                    "metadata": {
                        "dataset": self.dataset_name,
                        "split": split,
                        "sample_id": row.get("financebench_id") or row.get("id") or f"{split}_{idx}",
                    },
                }
                emitted += 1
                if max_samples_per_category is not None and emitted >= max_samples_per_category:
                    return
