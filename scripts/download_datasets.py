#!/usr/bin/env python3
"""
download_datasets.py
Download all evaluation datasets into structured folders under `data/`
Categories: OCR, Vision/Multimodal, RAG/Financial QA, Credit Risk
"""

import os
import urllib.request
import zipfile
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import login

# Optional: login if private datasets require HuggingFace token
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")  # reads from exported env variable
if hf_token is None:
    raise ValueError("Please set HUGGINGFACE_HUB_TOKEN environment variable before running the script")

login(token=hf_token)

# Base data folder
BASE_DIR = os.path.join(os.getcwd(), "data")
CATEGORIES = ["ocr", "vision", "rag", "credit_risk_pd", "credit_risk_sentiment", "credit_risk_memo_generator"]

# Create folder structure
for cat in CATEGORIES:
    os.makedirs(os.path.join(BASE_DIR, cat), exist_ok=True)

print("✓ Created folder structure under data/")

########################################
# OCR datasets
########################################
OCR_DATASETS = [
    ("SROIE", None, None),  # manual / Kaggle
    ("FUNSD", None, None),  # manual
]

print("\n=== Downloading OCR datasets ===")
for name, hf_name, hf_config in OCR_DATASETS:
    folder = os.path.join(BASE_DIR, "ocr", name)
    os.makedirs(folder, exist_ok=True)
    if hf_name:
        print(f"Downloading {name} from HuggingFace...")
        ds = load_dataset(hf_name, hf_config) if hf_config else load_dataset(hf_name)
        ds.save_to_disk(folder)
        print(f"Saved {name} to {folder}")
    else:
        print(f"Manual step required: {name}. Place files under {folder}")

########################################
# Vision / Multimodal datasets
########################################

print("\n=== Downloading Vision / Multimodal datasets ===")
# ------------------------------
# 1️⃣ MMMU (Selected Subjects Only)
# ------------------------------
print(get_dataset_config_names("MMMU/MMMU"))
# ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 
# 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 
# 'Clinical_Medicine', 'Computer_Science', 'Design', 
# 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 
# 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 
# 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 
# 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
MMMU_SUBJECTS = [
    "Finance",
    "Economics",
    "Math",
    "Accounting",
]
print("\n--- MMMU (Selected Subjects) ---")
for subject in MMMU_SUBJECTS:
    folder = os.path.join(BASE_DIR, "vision", f"MMMU_{subject}")
    os.makedirs(folder, exist_ok=True)
    print(f"Downloading MMMU subject: {subject} ...")

    ds = load_dataset("MMMU/MMMU", subject)
    ds.save_to_disk(folder)
    print(f"Saved MMMU-{subject} to {folder}")

VISION_DATASETS = [
    # ("MathVista", "AI4Math/MathVista", None),
    ("DocVQA", "lmms-lab/DocVQA", "DocVQA"),  # manual
    # ("TextVQA", "lmms-lab/textvqa", None),
    ("ChartQA", "HuggingFaceM4/ChartQA", None),
    # ("OCR-VQA", "howard-hou/OCR-VQA", None),
    # ("AI2D", "lmms-lab/ai2d", None),
    ("InfographicsVQA", "lmms-lab/DocVQA", "InfographicVQA"),  # manual
    # ("VisualMRC", "NTT-hil-insight/VisualMRC", None),
    # ("DUDE", None, None),  # manual
    ("OmniDocBench", "Quivr/OmniDocBench", "full_dataset"),
    # ("PlotQA", "achang/plot_qa", None),
]

print("\n=== Downloading Vision / Multimodal datasets ===")
for name, hf_name, hf_config in VISION_DATASETS:
    folder = os.path.join(BASE_DIR, "vision", name)
    os.makedirs(folder, exist_ok=True)
    if hf_name and hf_name.startswith("http"):  # direct URL downloads
        if name == "ChartQA":
            # example: chartqa jsonl
            print(f"Downloading ChartQA JSON...")
            urllib.request.urlretrieve(hf_name, os.path.join(folder, "chartqa.jsonl"))
            print(f"Saved ChartQA to {folder}")
        elif name == "TextVQA":
            zip_path = os.path.join(folder, "textvqa_images.zip")
            print(f"Downloading TextVQA images...")
            urllib.request.urlretrieve(hf_name, zip_path)
            print("Extracting TextVQA images...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(os.path.join(folder, "images"))
            print(f"Done extracting TextVQA images to {folder}")
    elif hf_name:  # HuggingFace datasets
        print(f"Downloading {name} from HuggingFace...")
        ds = load_dataset(hf_name, hf_config) if hf_config else load_dataset(hf_name)
        ds.save_to_disk(folder)
        print(f"Saved {name} to {folder}")
    else:
        print(f"Manual step required: {name}. Place files under {folder}")

########################################
# RAG / Financial QA datasets
########################################
RAG_DATASETS = [
    # ("HotpotQA", "hotpot_qa", "fullwiki"),
    ("FinQA", "FinanceMTEB/FinQA", None),
    ("TAT-QA", None, None),  # manual
    # ("BIRD-SQL", "domyn/FINCH", None),
]

print("\n=== Downloading RAG / Financial QA datasets ===")
for name, hf_name, hf_config in RAG_DATASETS:
    folder = os.path.join(BASE_DIR, "rag", name)
    os.makedirs(folder, exist_ok=True)
    if hf_name:
        print(f"Downloading {name} from HuggingFace...")
        ds = load_dataset(hf_name, hf_config) if hf_config else load_dataset(hf_name)
        ds.save_to_disk(folder)
        print(f"Saved {name} to {folder}")
    else:
        print(f"Manual step required: {name}. Place files under {folder}")

########################################
# Credit Risk datasets (PD)
########################################
CREDIT_RISK_PD_DATASETS = [
    # ("LendingClub", "https://resources.lendingclub.com/LoanStats3a.csv.zip"), # Official raw LendingClub CSV for feature engineering + training
    ("LendingClub", "TheFinAI/lendingclub-benchmark"), # TheFinAI HF benchmark for evaluation and comparison with others
]

print("\n=== Downloading Credit Risk (PD) datasets ===")
for name, source in CREDIT_RISK_PD_DATASETS:
    folder = os.path.join(BASE_DIR, "credit_risk_pd", name)
    os.makedirs(folder, exist_ok=True)
    if source.startswith("http"):  # direct download + unzip
        zip_path = os.path.join(folder, f"{name}.zip")
        print(f"Downloading {name} CSV/ZIP...")
        urllib.request.urlretrieve(source, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(folder)
        print(f"Done extracting {name} to {folder}")
    else:  # HuggingFace datasets
        print(f"Downloading {name} from HuggingFace...")
        ds = load_dataset(source)
        ds.save_to_disk(folder)
        print(f"Saved {name} to {folder}")

########################################
# Credit Risk datasets (Sentiment)
########################################
CREDIT_RISK_SENTIMENT_DATASETS = [
    ("FinancialPhraseBank", "FinanceMTEB/financial_phrasebank", None),
    # ("FinancialPhraseBank", "takala/financial_phrasebank", "sentences_allagree"),
    ("FiQA", "TheFinAI/fiqa-sentiment-classification", None),
]

print("\n=== Downloading Credit Risk (sentiment) datasets ===")
for name, source, hf_config in CREDIT_RISK_SENTIMENT_DATASETS:
    print(f"source: {source} hf_config: {hf_config}")
    folder = os.path.join(BASE_DIR, "credit_risk_sentiment", name)
    os.makedirs(folder, exist_ok=True)
    if source.startswith("http"):  # direct download + unzip
        zip_path = os.path.join(folder, f"{name}.zip")
        print(f"Downloading {name} CSV/ZIP...")
        urllib.request.urlretrieve(source, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(folder)
        print(f"Done extracting {name} to {folder}")
    else:  # HuggingFace datasets
        print(f"Downloading {name} from HuggingFace...")
        ds = load_dataset(source, hf_config) if hf_config else load_dataset(source)
        ds.save_to_disk(folder)
        print(f"Saved {name} to {folder}")

########################################
# Credit Risk datasets (Risk Memo Generation)
########################################
CREDIT_RISK_MEMO_GENERATOR_DATASETS = [
    ("FinanceBench", "PatronusAI/financebench"),
    # ("ECTSum", "mrSoul7766/ECTSum"),
]

print("\n=== Downloading Credit Risk (memo generator) datasets ===")
for name, source in CREDIT_RISK_MEMO_GENERATOR_DATASETS:
    folder = os.path.join(BASE_DIR, "credit_risk_memo_generator", name)
    os.makedirs(folder, exist_ok=True)
    if source.startswith("http"):  # direct download + unzip
        zip_path = os.path.join(folder, f"{name}.zip")
        print(f"Downloading {name} CSV/ZIP...")
        urllib.request.urlretrieve(source, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(folder)
        print(f"Done extracting {name} to {folder}")
    else:  # HuggingFace datasets
        print(f"Downloading {name} from HuggingFace...")
        ds = load_dataset(source)
        ds.save_to_disk(folder)
        print(f"Saved {name} to {folder}")


print("\n✓ All dataset download steps finished.")
print("Please check manual instructions for any datasets marked 'manual step required'.")
