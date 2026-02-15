#!/bin/bash

# Download Models Script
# Downloads PaddleOCR ONNX model and other required models

set -e

echo "=================================="
echo "Downloading Required Models"
echo "=================================="

# Create models directory
mkdir -p models

# Download PaddleOCR Detection Model (ONNX)
echo ""
echo "1. Downloading PaddleOCR Detection Model (ONNX)..."
echo "--------------------------------------------------"

PADDLEOCR_URL="https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_infer.tar"
PADDLEOCR_FILE="models/paddleocr_det.tar"

if [ -f "models/paddleocr_det.onnx" ]; then
    echo "✓ PaddleOCR model already exists"
else
    echo "Downloading from PaddleOCR repository..."
    wget -O "$PADDLEOCR_FILE" "$PADDLEOCR_URL" || {
        echo "⚠ Failed to download from official source"
        echo "Please download manually from:"
        echo "https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md"
        echo ""
        echo "For now, creating placeholder..."
        touch models/paddleocr_det.onnx
    }
    
    if [ -f "$PADDLEOCR_FILE" ]; then
        tar -xf "$PADDLEOCR_FILE" -C models/
        
        # Convert to ONNX (requires paddle2onnx)
        if command -v paddle2onnx &> /dev/null; then
            echo "Converting to ONNX..."
            paddle2onnx \
                --model_dir models/en_PP-OCRv4_det_infer \
                --model_filename inference.pdmodel \
                --params_filename inference.pdiparams \
                --save_file models/paddleocr_det.onnx \
                --opset_version 11
            echo "✓ Conversion complete"
        else
            echo "⚠ paddle2onnx not installed"
            echo "Install with: pip install paddle2onnx"
            echo "Creating placeholder for now..."
            touch models/paddleocr_det.onnx
        fi
        
        rm -f "$PADDLEOCR_FILE"
        rm -rf models/en_PP-OCRv4_det_infer
    fi
fi

echo ""
echo "=================================="
echo "Model Download Complete!"
echo "=================================="
echo ""
echo "Downloaded models:"
ls -lh models/
echo ""
echo "Note: If placeholder files were created, you'll need to:"
echo "1. Install paddle2onnx: pip install paddle2onnx"
echo "2. Run this script again to convert models to ONNX"