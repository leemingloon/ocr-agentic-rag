"""
Dataset adapters for loading and evaluating datasets in a standardized way.
Each adapter handles a specific dataset and implements the logic to load its data, 
parse it into a universal format, and evaluate predictions against ground truth.
The BaseDatasetAdapter defines the common interface and utilities for all datasets, 
while each specific dataset adapter (e.g., SROIEAdapter, FUNSDAdapter) implements 
the dataset-specific directory path, file type loading, and parsing data schema logic.
"""

import os
import json
import shutil
import pytesseract
import numpy as np
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset, load_from_disk
from scipy.optimize import linear_sum_assignment

BASE_DIR = os.path.join(os.getcwd(), "data")

DATA_STRUCTURE_JSON = os.path.join(BASE_DIR, "data_structure.json")

# generate data/data_structure.json mapping dataset_name -> {split_name: path_to_split}
from data.extract_folder_structure import extract_splits
extract_splits(BASE_DIR, output_json=DATA_STRUCTURE_JSON)

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
        self.root_dir = os.path.join(os.getcwd(), "data", self.dataset_name)
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
        """Delete downloaded dataset folder, only if data_source_from_hf_or_manual=True."""
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
    
    # def evaluate_ocr(self, gt_regions, pred_regions, iou_threshold=0.5):
    #     """
    #     Deprecated: use evaluate_all() instead for comprehensive evaluation.
    #     IoU-based text matching evaluation.
    #     """

    #     def iou(boxA, boxB):
    #         xA = max(boxA[0], boxB[0])
    #         yA = max(boxA[1], boxB[1])
    #         xB = min(boxA[2], boxB[2])
    #         yB = min(boxA[3], boxB[3])

    #         inter_area = max(0, xB - xA) * max(0, yB - yA)

    #         boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    #         boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    #         union = boxA_area + boxB_area - inter_area

    #         return inter_area / union if union > 0 else 0

    #     matched = 0

    #     for gt in gt_regions:
    #         for pred in pred_regions:
    #             if gt["text"] == pred["text"]:
    #                 if gt["bbox"] and pred["bbox"]:
    #                     if iou(gt["bbox"], pred["bbox"]) >= iou_threshold:
    #                         matched += 1
    #                         break

    #     precision = matched / len(pred_regions) if pred_regions else 0
    #     recall = matched / len(gt_regions) if gt_regions else 0

    #     return {
    #         "precision": precision,
    #         "recall": recall
    #     }

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
        "notes": "Arrow shard files under 'train' folder"
    },
    "val": {
        "format": "folder",
        "dataset_path": "data/vision/ChartQA/val",
        "ground_truth_path": None,
        "notes": "Arrow shard files under 'val' folder"
    },
    "test": {
        "format": "folder",
        "dataset_path": "data/vision/ChartQA/test",
        "ground_truth_path": None,
        "notes": "Arrow shard files under 'test' folder"
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

3. Single-folder dataset (OmniDocBench):
FILE_MAPPING = {
    "train": {
            "format": "folder",
            "dataset_path": "data/vision/OmniDocBench/train/",
            "ground_truth_path": None,
            "notes": "1 arrow shard file under 'train' folder"
    },
}

Loading logic should dynamically handle:
- 'folder': load all arrow/Parquet files under the folder
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
    # -----------------------------
    # Utility: Normalize Text
    # -----------------------------
    # def pytesseract_to_regions(
    #         image: np.ndarray, 
    #         ocr_engine: pytesseract
    #     ) -> List[Dict]:
    #     """
    #     Convert a page image to a unified 'regions' format using PyTesseract OCR.
        
    #     Args:
    #         image: np.ndarray, grayscale or BGR
    #         ocr_engine: instance of TesseractOCR

    #     Returns:
    #         List[Dict] with keys: 'text', 'bbox', 'confidence'
    #     """
    #     # Run Tesseract OCR with full page
    #     ocr_result = ocr_engine.recognize(image)

    #     # Extract word-level data using image_to_data
    #     data = pytesseract.image_to_data(
    #         ocr_engine._preprocess_roi(image),
    #         lang=ocr_engine.lang,
    #         config=ocr_engine.config,
    #         output_type=pytesseract.Output.DICT
    #     )

    #     regions = []
    #     for i, word in enumerate(data['text']):
    #         conf = float(data['conf'][i])
    #         if not word.strip() or conf <= 0:
    #             continue

    #         x, y, w, h = int(data['left'][i]), int(data['top'][i]), int(data['width'][i]), int(data['height'][i])
    #         region = {
    #             "text": word.strip(),
    #             "bbox": [x, y, x + w, y + h],
    #             "confidence": conf
    #         }
    #         regions.append(region)

    #     return regions

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
    
    def evaluate_all(self, gt_regions, pred_regions):
        results = {}
        results.update(
            self.evaluate_detection_recognition(gt_regions, pred_regions)
        )
        results.update(self.evaluate_cer(gt_regions, pred_regions))
        results.update(self.evaluate_wer(gt_regions, pred_regions))
        return results
    
    def evaluate_sample(self, gt_regions, pred_regions):
        return self.evaluate_all(gt_regions, pred_regions)

    def evaluate_dataset(self, gt_samples, pred_samples):
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
        if split not in file_mapping:
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

# ------------------------
# OCR Adapters
# ------------------------
"""
universal_format_for_OCR_datasets = {
    "image": Path,
    "regions": [
        {
            "text": str,
            "bbox": [x1, y1, x2, y2] | None,
            "label": str | None
        }
    ]
}
"""
class SROIEAdapter(OCRDatasetAdapter):
    """
    data/ocr/SROIE/SROIE2019/train/box/X00016469612.txt
    
    72,25,326,25,326,64,72,64,TAN WOON YANN
    50,82,440,82,440,121,50,121,BOOK TA .K(TAMAN DAYA) SDN BND
    205,121,285,121,285,139,205,139,789417-W
    110,144,383,144,383,163,110,163,NO.53 55,57 & 59, JALAN SAGU 18,
    192,169,299,169,299,187,192,187,TAMAN DAYA,
    162,193,334,193,334,211,162,211,81100 JOHOR BAHRU,
    217,216,275,216,275,233,217,233,JOHOR.
    50,342,279,342,279,359,50,359,DOCUMENT NO : TD01167104
    50,372,96,372,96,390,50,390,DATE:
    165,372,342,372,342,389,165,389,25/12/2018 8:13:39 PM
    48,396,117,396,117,415,48,415,CASHIER:
    164,397,215,397,215,413,164,413,MANIS
    49,423,122,423,122,440,49,440,MEMBER:
    191,460,298,460,298,476,191,476,CASH BILL
    30,508,121,508,121,523,30,523,CODE/DESC
    200,507,247,507,247,521,200,521,PRICE
    276,506,306,506,306,522,276,522,DISC
    374,507,441,507,441,521,374,521,AMOUNT
    69,531,102,531,102,550,69,550,QTY
    221,531,247,531,247,545,221,545,RM
    420,529,443,529,443,547,420,547,RM
    27,570,137,570,137,583,27,583,9556939040116
    159,570,396,570,396,584,159,584,KF MODELLING CLAY KIDDY FISH
    77,598,113,598,113,613,77,613,1 PC
    138,597,148,597,148,607,138,607,*
    202,597,245,597,245,612,202,612,9.000
    275,598,309,598,309,612,275,612,0.00
    411,596,443,596,443,613,411,613,9.00
    245,639,293,639,293,658,245,658,TOTAL:
    118,671,291,671,291,687,118,687,ROUR DING ADJUSTMENT:
    408,669,443,669,443,684,408,684,0.00
    86,704,292,704,292,723,86,723,ROUND D TOTAL (RM):
    401,703,443,703,443,719,401,719,9.00
    205,744,243,744,243,765,205,765,CASH
    402,748,441,748,441,763,402,763,10.00
    205,770,271,770,271,788,205,788,CHANGE
    412,772,443,772,443,786,412,786,1.00
    97,845,401,845,401,860,97,860,GOODS SOLD ARE NOT RETURNABLE OR
    190,864,309,864,309,880,190,880,EXCHANGEABLE
    142,883,353,883,353,901,142,901,***
    137,903,351,903,351,920,137,920,***
    202,942,292,942,292,959,202,959,THANK YOU
    163,962,330,962,330,977,163,977,PLEASE COME AGAIN !
    412,639,442,639,442,654,412,654,9.00
    
    data/ocr/SROIE/SROIE2019/train/entities/X00016469612.txt, 
    same file name and file type but under entities/ sub-folder, full raw data in X00016469612.txt:

    {
        "company": "BOOK TA .K (TAMAN DAYA) SDN BHD",
        "date": "25/12/2018",
        "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR.",
        "total": "9.00"
    }

    data/ocr/SROIE/SROIE2019/train/img/X00016469612.jpg,
    same file name but under entities sub-folder and as jpg file type.

    data/ocr/SROIE/SROIE2019/train/img/X00016469619.jpg is the next jpg file in the same img sub-folder,
    and so on for all samples in train/ and test/ sub-folders.
    The sub-folder structure is consistent across train and test split:
    data/ocr/SROIE/SROIE2019/test/box
    data/ocr/SROIE/SROIE2019/test/entities
    data/ocr/SROIE/SROIE2019/test/img
    data/ocr/SROIE/SROIE2019/train/box
    data/ocr/SROIE/SROIE2019/train/entities
    data/ocr/SROIE/SROIE2019/train/img

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under img/, box/, and entities/ folders.
    as well as as the pathing down to the individual sample level.
    The parsing logic for each sample is implemented in the _parse_sroie_sample() function, 
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """

    FILE_MAPPING = {
        "train": {
            "format": "folder",
            "dataset_path": "data/ocr/SROIE/SROIE2019/train",
            "ground_truth_path": None,
            "notes": "Images in img/, OCR boxes in box/, entities in entities/"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/ocr/SROIE/SROIE2019/test",
            "ground_truth_path": None,
            "notes": "Images in img/, OCR boxes in box/, entities in entities/"
        }
    }

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

    def load_split(self, dataset_split=None, max_samples_per_split=None, max_samples_per_category=None):
        """Load one split or all splits if dataset_split is None."""
        splits_to_load = [dataset_split or self.dataset_split] if dataset_split or self.dataset_split else list(self.FILE_MAPPING.keys())
        all_samples = []

        for split in splits_to_load:
            if split not in self.FILE_MAPPING:
                raise ValueError(f"SROIE supports splits: {list(self.FILE_MAPPING.keys())}")

            split_info = self.FILE_MAPPING[split]
            base_dir = Path.cwd() / split_info["dataset_path"]
            img_dir = base_dir / "img"
            box_dir = base_dir / "box"
            entity_dir = base_dir / "entities"

            for d in [img_dir, box_dir, entity_dir]:
                if not d.exists():
                    raise FileNotFoundError(d)

            samples = self._load_images_and_labels(
                img_folder=img_dir,
                box_folder=box_dir,
                entity_folder=entity_dir,
                parse_sample_fn=self._parse_sroie_sample,
                max_samples_per_split=max_samples_per_split
            )
            all_samples.extend(samples)

        if max_samples_per_category:
            all_samples = all_samples[:max_samples_per_category]

        return all_samples
    
    def _parse_sroie_sample(self, box_file, entity_file):
        """
        Parse SROIE sample for OCR evaluation.
        Ignores entities JSON (document-level labels).
        """

        regions = []

        if not box_file or not Path(box_file).exists():
            raise FileNotFoundError(f"Box file missing: {box_file}")

        with open(box_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(",")

            if len(parts) < 9:
                continue

            coords = list(map(int, parts[:8]))
            text = ",".join(parts[8:]).strip()

            xs = coords[0::2]
            ys = coords[1::2]

            bbox = [min(xs), min(ys), max(xs), max(ys)]

            regions.append({
                "text": text,
                "bbox": bbox,
                "label": None
            })

        return regions

class FUNSDAdapter(OCRDatasetAdapter):
    """
    below are 2 samples of the content in data/ocr/FUNSD/training_data/annotations/00040534.json:
    
    {
        "form": [
            {
                "box": [
                    84,
                    109,
                    136,
                    119
                ],
                "text": "COMPOUND",
                "label": "question",
                "words": [
                    {
                        "box": [
                            84,
                            109,
                            136,
                            119
                        ],
                        "text": "COMPOUND"
                    }
                ],
                "linking": [
                    [
                        0,
                        37
                    ]
                ],
                "id": 0
            },
            {
                "box": [
                    85,
                    141,
                    119,
                    152
                ],
                "text": "SOURCE",
                "label": "question",
                "words": [
                    {
                        "box": [
                            85,
                            141,
                            119,
                            152
                        ],
                        "text": "SOURCE"
                    }
                ],
                "linking": [
                    [
                        1,
                        38
                    ]
                ],
                "id": 1
            },
    
    data/ocr/FUNSD/training_data/images/00040534.png,
    same file name but under images sub-folder and as png file type.

    data/ocr/FUNSD/training_data/images/00070353.png is the next png file in the same images/ sub-folder,
    and so on for all samples in training_data/ and testing_data/ sub-folders.
    The sub-folder structure is consistent across train and test splits:
    data/ocr/FUNSD/training_data/annotations
    data/ocr/FUNSD/training_data/images
    data/ocr/FUNSD/testing_data/annotations
    data/ocr/FUNSD/testing_data/images

    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under annotations/ and images/ folders.
    The parsing logic for each sample is implemented in the _parse_funsd_annotation() function,
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    FILE_MAPPING = {
        "train": {
            "format": "folder",
            "dataset_path": "data/ocr/FUNSD/training_data",
            "ground_truth_path": None,
            "notes": "Images in images/, annotations in annotations/"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/ocr/FUNSD/testing_data",
            "ground_truth_path": None,
            "notes": "Images in images/, annotations in annotations/"
        }
    }

    def __init__(
        self,
        category: str = "ocr",
        dataset_name: str = "FUNSD",
        data_source_from_hf_or_manual: str = "manual",
        hf_repo_name: str = None,
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

    def load_split(self, dataset_split=None, max_samples_per_split=None, max_samples_per_category=None):
        """Load one split or all splits if dataset_split is None."""
        splits_to_load = [dataset_split or self.dataset_split] if dataset_split or self.dataset_split else list(self.FILE_MAPPING.keys())
        all_samples = []

        for split in splits_to_load:
            if split not in self.FILE_MAPPING:
                raise ValueError(f"FUNSD supports splits: {list(self.FILE_MAPPING.keys())}")

            split_info = self.FILE_MAPPING[split]
            base_dir = Path.cwd() / split_info["dataset_path"]
            annotations_dir = base_dir / "annotations"
            images_dir = base_dir / "images"

            for d in [annotations_dir, images_dir]:
                if not d.exists():
                    raise FileNotFoundError(f"Missing {d}")

            samples = self._load_images_and_labels(
                img_folder=images_dir,
                box_folder=annotations_dir,
                entity_folder=None,
                parse_sample_fn=self._parse_funsd_annotation,
                max_samples_per_split=max_samples_per_split,
                file_ext=".json"
            )
            all_samples.extend(samples)

        if max_samples_per_category:
            all_samples = all_samples[:max_samples_per_category]

        return all_samples

    def _parse_funsd_annotation(self, annotation_file, _=None):
        data = self._load_json(annotation_file)
        regions = []

        for item in data.get("form", []):
            label = item.get("label")
            for word in item.get("words", []):
                text = word.get("text", "").strip()
                box = word.get("box", [])
                if not text or not box:
                    continue

                x1, y1, x2, y2 = box
                regions.append({
                    "text": text,
                    "bbox": [x1, y1, x2, y2],
                    "label": label
                })

        return regions

# ===============================
# VisionDatasetAdapter (Intermediate Vision class)
# ===============================
class VisionDatasetAdapter(BaseDatasetAdapter):
    """
    Intermediate Vision adapter with dataset-agnostic loader helpers
    for folder-based (Arrow) or JSON-based Vision datasets.

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
    # Generic helper: load Arrow folder
    # -----------------------------------------
    def _load_arrow_folder(self, folder_path: Path, max_samples_per_split=None, max_samples_per_category=None):
        """
        Load HuggingFace Arrow shards inside a folder.

        Returns:
            List of raw samples (dict)
        """
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        from datasets import load_from_disk

        dataset = load_from_disk(str(folder_path))

        if max_samples_per_split:
            dataset = dataset.select(range(min(len(dataset), max_samples_per_split)))

        return list(dataset)

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
            - folder (Arrow shards)
            - json
            - json_with_gt
        """

        if split not in file_mapping:
            raise ValueError(f"Split '{split}' not in FILE_MAPPING")

        split_info = file_mapping[split]
        dataset_path = Path().cwd() / split_info["dataset_path"]

        fmt = split_info["format"]

        # -------------------------
        # Folder-based (Arrow)
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
universal_format_for_vision_datasets = {
    "input": ...,
    "ground_truth": ...,
    "metadata": {
        "dataset": ...,
        "sample_id": ...
    }
}
{
    "image": image_path_or_bytes_or_id,
    "question": str or None,
    "answer": str or None,
    "metadata": dict (optional)
}
"""
class DocVQAAdapter(VisionDatasetAdapter):
    """
    data/vision/DocVQA/test/data-00000-of-00008.arrow
    
    Schema or file content not available yet..
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_docvqa_annotation().
    
    data/vision/DocVQA/test/data-00001-of-00008.arrow,
    data/vision/DocVQA/test/data-00002-of-00008.arrow,
    data/vision/DocVQA/test/data-00003-of-00008.arrow,
    data/vision/DocVQA/test/data-00004-of-00008.arrow,
    data/vision/DocVQA/test/data-00005-of-00008.arrow,
    data/vision/DocVQA/test/data-00006-of-00008.arrow,
    data/vision/DocVQA/test/data-00007-of-00008.arrow,
    are the next remaining arrow shard files in the same test/ sub-folder.

    data/vision/DocVQA/validation/data-00000-of-00008.arrow,
    data/vision/DocVQA/validation/data-00001-of-00008.arrow,
    data/vision/DocVQA/validation/data-00002-of-00008.arrow,
    data/vision/DocVQA/validation/data-00003-of-00008.arrow,
    data/vision/DocVQA/validation/data-00004-of-00008.arrow,
    data/vision/DocVQA/validation/data-00005-of-00008.arrow,
    data/vision/DocVQA/validation/data-00006-of-00008.arrow,
    data/vision/DocVQA/validation/data-00007-of-00008.arrow,
    are the arrow shard files in the validation/ sub-folder.

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
    FILE_MAPPING = {
        "test": {
            "format": "folder",
            "dataset_path": "data/vision/DocVQA/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 8 Arrow shard files"
        },
        "validation": {
            "format": "folder",
            "dataset_path": "data/vision/DocVQA/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 8 Arrow shard files"
        },
    }

    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "DocVQA",
        data_source_from_hf_or_manual: str = "manual",
        hf_repo_name: str = None,
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
    
    def load_split(self, dataset_split=None, max_samples_per_split=None, max_samples_per_category=None):
        splits_to_load = [dataset_split or self.dataset_split] if dataset_split or self.dataset_split else list(self.FILE_MAPPING.keys())
        all_samples = []

        for split in splits_to_load:
            raw_samples = self.vision_dataset_loader(split, self.FILE_MAPPING, max_samples_per_split=max_samples_per_split)

            # standardize to universal_format_for_vision_datasets
            for idx, s in enumerate(raw_samples):
                all_samples.append({
                    "input": {
                        "image": s.get("image"),
                        "question": s.get("question")
                    },
                    "ground_truth": s.get("answers")[0] if s.get("answers") else None,
                    "metadata": {"dataset": "DocVQA", "sample_id": idx}
                })

        return all_samples

class InfographicsVQAAdapter(VisionDatasetAdapter):
    """
    data/vision/InfographicsVQA/test/data-00000-of-00004.arrow
    
    Schema or file content not available yet.. 
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_infographicsvqa_annotation().
    
    data/vision/InfographicsVQA/test/data-00001-of-00004.arrow,
    data/vision/InfographicsVQA/test/data-00002-of-00004.arrow,
    data/vision/InfographicsVQA/test/data-00003-of-00004.arrow,
    are the next remaining arrow shard files in the same test/ sub-folder.

    data/vision/InfographicsVQA/validation/data-00000-of-00004.arrow,
    data/vision/InfographicsVQA/validation/data-00001-of-00004.arrow,
    data/vision/InfographicsVQA/validation/data-00002-of-00004.arrow,
    data/vision/InfographicsVQA/validation/data-00003-of-00004.arrow,
    are the arrow shard files in the validation/ sub-folder.

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
    FILE_MAPPING = {
        "test": {
            "format": "folder",
            "dataset_path": "data/vision/InfographicsVQA/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 4 Arrow shard files"
        },
        "validation": {
            "format": "json",
            "dataset_path": "data/vision/InfographicsVQA/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 4 Arrow shard files"
        },
    }


    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "InfographicsVQA",
        data_source_from_hf_or_manual: str = "manual",
        hf_repo_name: str = None,
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
    
    def load_split(self, dataset_split=None, max_samples_per_split=None, max_samples_per_category=None):
        splits_to_load = [dataset_split or self.dataset_split] if dataset_split or self.dataset_split else list(self.FILE_MAPPING.keys())
        all_samples = []

        for split in splits_to_load:
            raw_samples = self.vision_dataset_loader(split, self.FILE_MAPPING, max_samples_per_split=max_samples_per_split)

            for idx, s in enumerate(raw_samples):
                all_samples.append({
                    "input": {
                        "image": s.get("image"),
                        "question": s.get("question")
                    },
                    "ground_truth": s.get("answers")[0] if s.get("answers") else None,
                    "metadata": {"dataset": "InfographicsVQA", "sample_id": idx}
                })

        return all_samples

class ChartQAAdapter(VisionDatasetAdapter):
    """
    data/vision/ChartQA/test/data-00000-of-00001.arrow,
    is the only arrow shard file in test/ sub-folder.
    
    Schema or file content not available yet.. 
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_chartqa_annotation().
    
    data/vision/ChartQA/train/data-00000-of-00003.arrow,
    data/vision/ChartQA/train/data-00001-of-00003.arrow,
    data/vision/ChartQA/train/data-00002-of-00003.arrow,
    are the 3 arrow shard files in train/ sub-folder.

    data/vision/ChartQA/val/data-00000-of-00001.arrow,
    is the only arrow shard file in val/ sub-folder.

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
            "notes": "Test folder containing 1 Arrow shard file"
        },
        "train": {
            "format": "folder",
            "dataset_path": "data/vision/ChartQA/train",
            "ground_truth_path": None,
            "notes": "Train folder containing 3 Arrow shard files"
        },
        "val": {
            "format": "folder",
            "dataset_path": "data/vision/ChartQA/val",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Arrow shard file"
        },
    }


    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "ChartQA",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = None,
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

    def load_split(self, dataset_split=None, max_samples_per_split=None, max_samples_per_category=None):
        splits_to_load = [dataset_split or self.dataset_split] if dataset_split or self.dataset_split else list(self.FILE_MAPPING.keys())
        all_samples = []

        for split in splits_to_load:
            raw_samples = self.vision_dataset_loader(split, self.FILE_MAPPING, max_samples_per_split=max_samples_per_split)

            for idx, row in enumerate(raw_samples):
                all_samples.append({
                    "input": {"image": row.get("image", row)},
                    "ground_truth": row.get("answer"),
                    "metadata": {"dataset": "ChartQA", "sample_id": idx}
                })

        return all_samples

class OmniDocBenchAdapter(VisionDatasetAdapter):
    """
    data/vision/OmniDocBench/train/data-00000-of-00001.arrow,
    is the only arrow shard file in train/ sub-folder.
    
    Schema or file content not available yet.. 
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_omnidocbench_annotation().

        
    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under train/ folder.
    The parsing logic for each sample is implemented in the _parse_omnidocbench_annotation() function,
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    FILE_MAPPING = {
        "default": {
            "format": "folder",
            "dataset_path": "data/vision/OmniDocBench/train",
            "ground_truth_path": None,
            "notes": "Single arrow file under 'train'; no official splits"
        }
    }

    def __init__(
        self,
        category: str = "vision",
        dataset_name: str = "OmniDocBench",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = None,
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

    def load_split(self, dataset_split=None, max_samples_per_split=None, max_samples_per_category=None, recursive=True):
        split = dataset_split or self.dataset_split
        raw_samples = self.vision_dataset_loader(split, self.FILE_MAPPING, max_samples_per_split=max_samples_per_split)

        # For OmniDocBench we just return files (Arrow folder)
        standardized = []
        for idx, s in enumerate(raw_samples):
            standardized.append({
                "input": s,
                "ground_truth": None,
                "metadata": {"dataset": "OmniDocBench", "sample_id": idx}
            })
        return standardized

class MMMUAccountingAdapter(VisionDatasetAdapter):
    """
    data/vision/MMMU_Accounting/dev/data-00000-of-00001.arrow,
    is the only arrow shard file in dev/ sub-folder.
    
    Schema or file content not available yet.. 
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_mmmu_accounting_annotation().
    
    data/vision/MMMU_Accounting/test/data-00000-of-00001.arrow,
    is the only arrow shard file in test/ sub-folder.

    data/vision/MMMU_Accounting/validation/data-00000-of-00001.arrow,
    is the only arrow shard file in validation/ sub-folder.

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
    FILE_MAPPING = {
        "dev": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Accounting/dev",
            "ground_truth_path": None,
            "notes": "Dev folder containing 1 Arrow shard file"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Accounting/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Arrow shard file"
        },
        "validation": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Accounting/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Arrow shard file"
        },
    }
    pass

class MMMUEconomicsAdapter(VisionDatasetAdapter):
    """
    data/vision/MMMU_Economics/dev/data-00000-of-00001.arrow,
    is the only arrow shard file in dev/ sub-folder.
    
    Schema or file content not available yet..
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_mmmu_economics_annotation().
    
    data/vision/MMMU_Economics/test/data-00000-of-00001.arrow,
    is the only arrow shard file in test/ sub-folder.

    data/vision/MMMU_Economics/validation/data-00000-of-00001.arrow,
    is the only arrow shard file in validation/ sub-folder.

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
    FILE_MAPPING = {
        "dev": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Economics/dev",
            "ground_truth_path": None,
            "notes": "Dev folder containing 1 Arrow shard file"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Economics/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Arrow shard file"
        },
        "validation": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Economics/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Arrow shard file"
        },
    }
    pass

class MMMUFinanceAdapter(VisionDatasetAdapter):
    """
    data/vision/MMMU_Finance/dev/data-00000-of-00001.arrow,
    is the only arrow shard file in dev/ sub-folder.
    
    Schema or file content not available yet.. 
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_mmmu_finance_annotation().
    
    data/vision/MMMU_Finance/test/data-00000-of-00001.arrow,
    is the only arrow shard file in test/ sub-folder.

    data/vision/MMMU_Finance/validation/data-00000-of-00001.arrow,
    is the only arrow shard file in validation/ sub-folder.

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
    FILE_MAPPING = {
        "dev": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Finance/dev",
            "ground_truth_path": None,
            "notes": "Dev folder containing 1 Arrow shard file"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Finance/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Arrow shard file"
        },
        "validation": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Finance/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Arrow shard file"
        },
    }
    pass

class MMMUMathAdapter(VisionDatasetAdapter):
    """
    data/vision/MMMU_Math/dev/data-00000-of-00001.arrow,
    is the only arrow shard file in dev/ sub-folder.
    
    Schema or file content not available yet.. 
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_mmmu_math_annotation().
    
    data/vision/MMMU_Math/test/data-00000-of-00001.arrow,
    is the only arrow shard file in test/ sub-folder.

    data/vision/MMMU_Math/validation/data-00000-of-00001.arrow,
    is the only arrow shard file in validation/ sub-folder.

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
    FILE_MAPPING = {
        "dev": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Math/dev",
            "ground_truth_path": None,
            "notes": "Dev folder containing 1 Arrow shard file"
        },
        "test": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Math/test",
            "ground_truth_path": None,
            "notes": "Test folder containing 1 Arrow shard file"
        },
        "validation": {
            "format": "folder",
            "dataset_path": "data/vision/MMMU_Math/validation",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Arrow shard file"
        },
    }
    pass

# class DUDEAdapter(VisionDatasetAdapter):
#     FILE_MAPPING = {
#         "default": {
#             "format": "json_with_gt",
#             "dataset_path": "data/vision/DUDE/DUDE_sample_dataset.json",
#             "ground_truth_path": "data/vision/DUDE/DUDE_dataset-sample_gt.json",
#             "notes": "No standard splits; evaluation dataset + ground truth file"
#         }
#     }

#     def __init__(
#         self,
#         category: str = "vision",
#         dataset_name: str = "DUDE",
#         data_source_from_hf_or_manual: str = "manual",
#         hf_repo_name: str = None,
#         hf_repo_variant: str = None,
#         dataset_split=None
#     ):
#         super().__init__(
#             category=category,
#             dataset_name=dataset_name,
#             data_source_from_hf_or_manual=data_source_from_hf_or_manual,
#             hf_repo_name=hf_repo_name,
#             hf_repo_variant=hf_repo_variant
#         )
#         self.category = category
#         self.dataset_name = dataset_name
#         self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
#         self.hf_repo_name = hf_repo_name
#         self.hf_repo_variant = hf_repo_variant
#         self.dataset_split = dataset_split
#         self.FILE_MAPPING = self.FILE_MAPPING

#     def load_split(self, dataset_split="val", max_samples_per_split=None, max_samples_per_category=None):
#         if dataset_split != "val":
#             raise ValueError("DUDE only supports 'val' for evaluation.")

#         raw_samples = self.vision_dataset_loader("default", self.FILE_MAPPING, max_samples_per_split=max_samples_per_split)

#         standardized = []
#         for idx, s in enumerate(raw_samples):
#             standardized.append({
#                 "input": s,
#                 "ground_truth": s.get("answer"),
#                 "metadata": {"dataset": "DUDE", "sample_id": idx}
#             })
#         return standardized

# class AI2DAdapter(VisionDatasetAdapter):
#     FILE_MAPPING = {
#         "test": {
#             "format": "folder",
#             "dataset_path": "data/vision/AI2D/test",
#             "ground_truth_path": None,
#             "notes": "Only test split available; contains 2 arrow shard files"
#         }
#     }

#     def __init__(
#         self,
#         category: str = "vision",
#         dataset_name: str = "AI2D",
#         data_source_from_hf_or_manual: str = "hf",
#         hf_repo_name: str = None,
#         hf_repo_variant: str = None,
#         dataset_split=None
#     ):
#         super().__init__(
#             category=category,
#             dataset_name=dataset_name,
#             data_source_from_hf_or_manual=data_source_from_hf_or_manual,
#             hf_repo_name=hf_repo_name,
#             hf_repo_variant=hf_repo_variant
#         )
#         self.category = category
#         self.dataset_name = dataset_name
#         self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
#         self.hf_repo_name = hf_repo_name
#         self.hf_repo_variant = hf_repo_variant
#         self.dataset_split = dataset_split
#         self.FILE_MAPPING = self.FILE_MAPPING

#     def load_split(self, dataset_split=None, max_samples_per_split=None, max_samples_per_category=None):
#         splits_to_load = [dataset_split or self.dataset_split] if dataset_split or self.dataset_split else list(self.FILE_MAPPING.keys())
#         all_samples = []

#         for split in splits_to_load:
#             raw_samples = self.vision_dataset_loader(split, self.FILE_MAPPING, max_samples_per_split=max_samples_per_split)

#             for idx, row in enumerate(raw_samples):
#                 all_samples.append({
#                     "input": {"image": row.get("image", row)},
#                     "ground_truth": row.get("answer"),
#                     "metadata": {"dataset": "AI2D", "sample_id": idx}
#                 })

#         return all_samples

# ------------------------
# RAG Adapters
# ------------------------
# class HotpotQAAdapter(BaseDatasetAdapter):
#     FILE_MAPPING = {
#         "train": {
#             "format": "folder",
#             "dataset_path": "data/rag/HotpotQA/train",
#             "ground_truth_path": None,
#             "notes": "2 arrow shard files under 'train' folder"
#         },
#         "val": {
#             "format": "folder",
#             "dataset_path": "data/rag/HotpotQA/validation",
#             "ground_truth_path": None,
#             "notes": "1 arrow file under 'validation' folder"
#         },
#         "test": {
#             "format": "folder",
#             "dataset_path": "data/rag/HotpotQA/test",
#             "ground_truth_path": None,
#             "notes": "1 arrow file under 'test' folder"
#         }
#     }

#     def __init__(
#         self,
#         category: str = "rag",
#         dataset_name: str = "HotpotQA",
#         data_source_from_hf_or_manual: str = "hf",
#         hf_repo_name: str = None,
#         hf_repo_variant: str = None,
#         dataset_split=None
#     ):
#         super().__init__(
#             category=category,
#             dataset_name=dataset_name,
#             data_source_from_hf_or_manual=data_source_from_hf_or_manual,
#             hf_repo_name=hf_repo_name,
#             hf_repo_variant=hf_repo_variant
#         )
#         self.category = category
#         self.dataset_name = dataset_name
#         self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
#         self.hf_repo_name = hf_repo_name
#         self.hf_repo_variant = hf_repo_variant
#         self.dataset_split = dataset_split
#         self.FILE_MAPPING = self.FILE_MAPPING

#     def load_split(self, dataset_split=None, recursive=False, max_samples_per_split=None, max_samples_per_category=None):
#         dataset_split = dataset_split or self.dataset_split
#         self.download()
#         def transform(row):
#             context = " ".join([" ".join(p[1]) for p in row.get("context", [])])
#             return {"question": row.get("question"), "context": context}
#         return self._prepare_samples(
#             dataset_split_obj=self.dataset_obj.get(dataset_split, []),
#             dataset_name="HotpotQA",
#             max_samples_per_split=max_samples_per_split,
#             input_keys=None,
#             answer_key="answer",
#             transform=transform
#         )

# ------------------------
# RAG Adapters
# ------------------------
class FinQAAdapter(BaseDatasetAdapter):
    """
    data/rag/FinQA/train/data-00000-of-00001.arrow,
    is the only arrow shard file in train/ sub-folder.
    
    Schema or file content not available yet..
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_finqa_annotation().

    
    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under train/ folder.
    The parsing logic for each sample is implemented in the _parse_finqa_annotation() function,
    which is called by the generic _load_samples() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    FILE_MAPPING = {
        "train": {
            "format": "folder",
            "dataset_path": "data/rag/FinQA/train",
            "ground_truth_path": None,
            "notes": "1 arrow file under 'train' folder"
        }
    }

    def __init__(
        self,
        category: str = "rag",
        dataset_name: str = "FinQA",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = None,
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

    def load_split(self, dataset_split=None, recursive=False, max_samples_per_split=None, max_samples_per_category=None):
        dataset_split = dataset_split or self.dataset_split
        self.download()
        return self._prepare_samples(
            dataset_split_obj=self.dataset_obj.get(dataset_split, []),
            dataset_name="FinQA",
            max_samples_per_split=max_samples_per_split,
            input_keys=["question", "table", "text"],
            answer_key="answer"
        )

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


    FILE_MAPPING below only serves to provide the path right down to the split level.
    For each split, the actual loading logic in load_split() must handle the pathing,
    down to the individual sample level under TAT-QA/ folder.
    as well as as the pathing down to the individual sample level.
    The parsing logic for each sample is implemented in the _parse_tatqa_sample() function, 
    which is called by the generic _load_images_and_labels() helper,
    and serves to extract structured data based on a schema shared across all samples within the same file type.
    """
    FILE_MAPPING = {
        "dev": {
            "format": "json",
            "dataset_path": "data/rag/TAT-QA/dev.json",
            "ground_truth_path": None,
            "notes": None
        },
        "test": {
            "format": "json",
            "dataset_path": "data/rag/TAT-QA/test.json",
            "ground_truth_path": "data/rag/TAT-QA/test_gold.json",
            "notes": "2 json files: Ground truth for test set is in 'test_gold.json', test set is in 'test.json'"
        },
        "train": {
            "format": "json",
            "dataset_path": "data/rag/TAT-QA/train.json",
            "ground_truth_path": None,
            "notes": None
        },
    }


    def __init__(
        self,
        category: str = "rag",
        dataset_name: str = "TATQA",
        data_source_from_hf_or_manual: str = "manual",
        hf_repo_name: str = None,
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

    def load_split(self, dataset_split="dev", recursive=False, max_samples_per_split=None, max_samples_per_category=None):
        file_map = {
            "train": "tatqa_dataset_train.json",
            "dev": "tatqa_dataset_dev.json",
            "test": "tatqa_dataset_test_gold.json"
        }
        file_name = file_map.get(dataset_split, file_map["dev"])
        path = os.path.join(self.root_dir, file_name)
        if not os.path.exists(path):
            print(f"⚠️ {self.dataset_name} {dataset_split} split not found")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def transform(row):
            return {"question": row.get("question"), "context": row.get("paragraphs")}

        return self._prepare_samples(
            dataset_split_obj=data,
            dataset_name="TATQA",
            max_samples_per_split=max_samples_per_split,
            input_keys=None,
            answer_key="answer",
            transform=transform
        )

# ------------------------
# Credit Risk Adapters (PD)
# ------------------------
class LendingClubAdapter(BaseDatasetAdapter):
    """
    data/credit_risk_pd/LendingClub/test/data-00000-of-00001.arrow,
    is the only arrow shard file in test/ sub-folder.
    
    Schema or file content not available yet.. 
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_lendingclub_annotation().
    
    data/credit_risk_pd/LendingClub/train/data-00000-of-00001.arrow,
    is the only arrow shard file in train/ sub-folder.

    data/credit_risk_pd/LendingClub/valid/data-00000-of-00001.arrow,
    is the only arrow shard file in valid/ sub-folder.

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
            "notes": "Test folder containing 1 Arrow shard file"
        },
        "train": {
            "format": "folder",
            "dataset_path": "data/credit_risk_pd/LendingClub/train",
            "ground_truth_path": None,
            "notes": "Train folder containing 1 Arrow shard file"
        },
        "valid": {
            "format": "folder",
            "dataset_path": "data/credit_risk_pd/LendingClub/valid",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Arrow shard file"
        },
    }
    def __init__(
        self,
        category: str = "credit_risk",
        dataset_name: str = "LendingClub",
        data_source_from_hf_or_manual: str = "manual",
        hf_repo_name: str = None,
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

    def load_split(self, dataset_split="train", recursive=False, max_samples_per_split=None, max_samples_per_category=None):
        csv_files = [f for f in os.listdir(self.root_dir) if f.endswith(".csv")]
        if not csv_files:
            print(f"⚠️ {self.dataset_name} CSV not found")
            return []

        import pandas as pd
        df = pd.concat([pd.read_csv(os.path.join(self.root_dir, f)) for f in csv_files], ignore_index=True)

        def transform(row):
            return row.to_dict()

        return self._prepare_samples(
            dataset_split_obj=df.itertuples(index=False),
            dataset_name="LendingClub",
            max_samples_per_split=max_samples_per_split,
            input_keys=None,
            answer_key="label",
            transform=transform
        )

# ------------------------
# Credit Risk Adapters (Sentiment)
# ------------------------

class FinancialPhraseBankAdapter(BaseDatasetAdapter):
    """
    data/credit_risk_sentiment/FinancialPhraseBank/test/data-00000-of-00001.arrow,
    is the only arrow shard file in test/ sub-folder.
    
    Schema or file content not available yet..
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_financial_phrase_bank_annotation().
    
    data/credit_risk_sentiment/FinancialPhraseBank/train/data-00000-of-00001.arrow,
    is the only arrow shard file in train/ sub-folder.

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
            "notes": "Test folder containing 1 Arrow shard file"
        },
        "train": {
            "format": "folder",
            "dataset_path": "data/credit_risk_sentiment/FinancialPhraseBank/train",
            "ground_truth_path": None,
            "notes": "Train folder containing 1 Arrow shard file"
        },
    }
    pass

class FiQAAdapter(BaseDatasetAdapter):
    """
    data/credit_risk_sentiment/FiQA/test/data-00000-of-00001.arrow,
    is the only arrow shard file in test/ sub-folder.
    
    Schema or file content not available yet.. 
    we need to load it once and print out the first 5 rows/set of samples to
    inspect the actual structure to implement the parsing logic in _parse_fiqa_annotation().
    
    data/credit_risk_sentiment/FiQA/train/data-00000-of-00001.arrow,
    is the only arrow shard file in train/ sub-folder.

    data/credit_risk_sentiment/FiQA/valid/data-00000-of-00001.arrow,
    is the only arrow shard file in valid/ sub-folder.

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
            "notes": "Test folder containing 1 Arrow shard file"
        },
        "train": {
            "format": "folder",
            "dataset_path": "data/credit_risk_sentiment/FiQA/train",
            "ground_truth_path": None,
            "notes": "Train folder containing 1 Arrow shard file"
        },
        "valid": {
            "format": "folder",
            "dataset_path": "data/credit_risk_sentiment/FiQA/valid",
            "ground_truth_path": None,
            "notes": "Validation folder containing 1 Arrow shard file"
        },
    }
    def __init__(
        self,
        category: str = "credit_risk",
        dataset_name: str = "FiQA",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = None,
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

    def load_split(self, dataset_split=None, recursive=False, max_samples_per_split=None, max_samples_per_category=None):
        dataset_split = dataset_split or self.dataset_split
        self.download()
        return self._prepare_samples(
            dataset_split_obj=self.dataset_obj.get(dataset_split, []),
            dataset_name="FiQA",
            max_samples_per_split=max_samples_per_split,
            input_keys=["sentence"],
            answer_key="label"
        )

# ------------------------
# Credit Risk Adapters (Memo Generator)
# ------------------------

class FinanceBenchAdapter(BaseDatasetAdapter):
    """
    data/credit_risk_memo_generator/FinanceBench/train/data-00000-of-00001.arrow,
    is the only arrow shard file in train/ sub-folder.
    
    Schema or file content not available yet.. 
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
            "notes": "Single arrow file under 'train'; no official splits"
        }
    }
    def __init__(
        self,
        category: str = "credit_risk",
        dataset_name: str = "FinanceBench",
        data_source_from_hf_or_manual: str = "hf",
        hf_repo_name: str = None,
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

    def load_split(self, dataset_split=None, recursive=False, max_samples_per_split=None, max_samples_per_category=None):
        dataset_split = dataset_split or self.dataset_split
        self.download()
        return self._prepare_samples(
            dataset_split_obj=self.dataset_obj.get(dataset_split, []),
            dataset_name="FinanceBench",
            max_samples_per_split=max_samples_per_split,
            input_keys=["question", "context"],
            answer_key="answer"
        )

# class ECTSumAdapter(BaseDatasetAdapter):
#     def __init__(
#         self,
#         category: str = "credit_risk",
#         dataset_name: str = "ECTSum",
#         data_source_from_hf_or_manual: str = "hf",
#         hf_repo_name: str = None,
#         hf_repo_variant: str = None,
#         dataset_split=None
#     ):
#         super().__init__(
#             category=category,
#             dataset_name=dataset_name,
#             data_source_from_hf_or_manual=data_source_from_hf_or_manual,
#             hf_repo_name=hf_repo_name,
#             hf_repo_variant=hf_repo_variant
#         )
#         self.category = category
#         self.dataset_name = dataset_name
#         self.data_source_from_hf_or_manual = data_source_from_hf_or_manual
#         self.hf_repo_name = hf_repo_name
#         self.hf_repo_variant = hf_repo_variant
#         self.dataset_split = dataset_split
#         self.FILE_MAPPING = self.FILE_MAPPING

#     def load_split(self, dataset_split=None, recursive=False, max_samples_per_split=None, max_samples_per_category=None):
#         dataset_split = dataset_split or self.dataset_split
#         self.download()
#         return self._prepare_samples(
#             dataset_split_obj=self.dataset_obj.get(dataset_split, []),
#             dataset_name="ECTSum",
#             max_samples_per_split=max_samples_per_split,
#             input_keys=["transcript"],
#             answer_key="summary"
#         )
