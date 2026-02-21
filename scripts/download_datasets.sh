echo "Downloading all datasets via download_datasets.py..."
python scripts/download_datasets.py

echo "Downloading SROIE (requires Kaggle API credentials)..."
kaggle datasets download -d urbikn/sroie-datasetv2 -p data/ocr/SROIE --unzip

echo "Other datasets marked 'manual step required' need to be placed manually in their folders."
