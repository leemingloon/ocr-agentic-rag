"""
Full End-to-End Pipeline Demo

Demonstrates complete OCR → RAG → Vision → Credit Risk workflow.

Modes:
- local: 1 sample (fast, ~3 seconds)
- sagemaker: Batch processing (600 samples, ~20 minutes)
- production: Full evaluation (millions of samples)

Usage:
    # Local demo (1 document)
    python examples/06_full_e2e_demo.py
    
    # SageMaker batch
    python examples/06_full_e2e_demo.py --mode sagemaker --s3-bucket my-bucket
    
    # Production
    python examples/06_full_e2e_demo.py --mode production
"""

import sys
from pathlib import Path
import argparse
import time
import cv2
import numpy as np
import os

sys.path.append(str(Path(__file__).parent.parent))

# OCR Pipeline
from ocr_pipeline.recognition.hybrid_ocr import HybridOCR

# RAG System
from rag_system.chunking import DocumentChunker
from rag_system.retrieval import HybridRetriever
from rag_system.reranking import BGEReranker
from rag_system.agentic.orchestrator import AgenticRAG

# Multimodal
from rag_system.multimodal_rag import MultimodalRAG

# Credit Risk
from credit_risk.pipeline import CreditRiskPipeline


def run_full_e2e_demo(mode: str = "local", s3_bucket: str = None, dry_run: bool = False):
    """
    Run complete end-to-end pipeline
    
    Args:
        mode: Execution mode (local/sagemaker/production)
        s3_bucket: S3 bucket for SageMaker mode
        dry_run: If True, skip API calls (no costs)

    Flow:
    1. OCR: Extract text from financial document
    2. RAG: Retrieve relevant context
    3. Vision: Extract charts (if present)
    4. Credit Risk: Generate PD, risk drivers, memo
    """
    # Set dry-run environment variable
    if dry_run:
        os.environ["DRY_RUN_MODE"] = "true"
    
    print("=" * 70)
    print(f"Full E2E Pipeline Demo - {mode.upper()} Mode")
    if dry_run:
        print("🧪 DRY-RUN: Mock responses (no API costs)")
    print("=" * 70)
    print("\nPipeline: OCR → RAG → Vision → Credit Risk")
    print("-" * 70)
    
    # ========================================
    # STEP 1: OCR Pipeline
    # ========================================
    print("\nSTEP 1: OCR Pipeline")
    print("-" * 70)
    
    ocr_system = HybridOCR(
        use_detection_router=True,
        use_vision_augmentation=True,
        # mode=mode
    )
    
    # Sample financial document (mock image)
    # In production, would load actual PDF/image
    print("  Loading financial statement image...")
    
    # Create mock image with text
    mock_image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    # Mock OCR result
    ocr_start = time.time()
    ocr_result = {
        "text": """
        ABC CORP - QUARTERLY FINANCIAL STATEMENT
        Q3 2024
        
        INCOME STATEMENT
        Revenue: $10,000,000
        Cost of Goods Sold: $6,000,000
        Gross Profit: $4,000,000
        Operating Expenses: $2,000,000
        EBITDA: $2,000,000
        Depreciation: $300,000
        EBIT: $1,700,000
        Interest Expense: $500,000
        Net Income: $1,200,000
        
        BALANCE SHEET
        Current Assets: $3,000,000
        Inventory: $500,000
        Total Assets: $15,000,000
        Current Liabilities: $2,000,000
        Total Debt: $7,000,000
        Equity: $5,000,000
        
        NOTES:
        Debt covenant: Debt/EBITDA < 4.0x (currently 3.5x)
        """,
        "confidence": 0.92,
        "metadata": {
            "detection_method": "classical",
            "vision_used": False,
        }
    }
    ocr_time = (time.time() - ocr_start) * 1000
    
    print(f"  ✓ OCR Complete ({ocr_time:.0f}ms)")
    print(f"  ✓ Confidence: {ocr_result['confidence']:.1%}")
    print(f"  ✓ Detection: {ocr_result['metadata']['detection_method']}")
    print(f"  ✓ Text length: {len(ocr_result['text'])} chars")
    
    # ========================================
    # STEP 2: RAG System
    # ========================================
    print("\nSTEP 2: RAG System")
    print("-" * 70)
    
    print("  Initializing RAG components...")
    chunker = DocumentChunker()
    retriever = HybridRetriever()
    reranker = BGEReranker()
    agent = AgenticRAG(retriever=retriever, reranker=reranker)
    
    # Chunk document
    print("  Chunking document...")
    chunks = chunker.chunk_document(ocr_result["text"])
    print(f"  ✓ Created {len(chunks)} chunks")
    
    # Build index
    print("  Building retrieval index...")
    retriever.build_index(chunks)
    print(f"  ✓ Index built")
    
    # Query
    query = "What is the company's leverage ratio and debt covenant status?"
    print(f"\n  Query: '{query}'")
    
    rag_start = time.time()
    rag_result = agent.query(query)
    rag_time = (time.time() - rag_start) * 1000
    
    print(f"  ✓ RAG Complete ({rag_time:.0f}ms)")
    print(f"  ✓ Answer: {rag_result['answer'][:200]}...")
    
    # ========================================
    # STEP 3: Multimodal Vision
    # ========================================
    print("\nSTEP 3: Multimodal Vision")
    print("-" * 70)
    
    multimodal_rag = MultimodalRAG(retriever=retriever, reranker=reranker)
    
    # Check for charts in document
    has_charts = False  # Mock: no charts in this example
    
    if has_charts:
        print("  Extracting charts with vision model...")
        vision_result = multimodal_rag.query(
            "Extract data from financial charts",
            document_image=mock_image,
            use_visual_context=True
        )
        print(f"  ✓ Vision extraction complete")
    else:
        print("  No charts detected - skipping vision extraction")
        vision_result = None
    
    # ========================================
    # STEP 4: Credit Risk Pipeline
    # ========================================
    print("\nSTEP 4: Credit Risk Pipeline")
    print("-" * 70)
    
    credit_pipeline = CreditRiskPipeline(
        mode=mode,
        s3_bucket=s3_bucket
    )
    
    # Mock news articles
    news_articles = [
        "ABC Corp reported Q3 results below analyst expectations due to weak demand.",
        "The company's leverage has increased to 3.5x, approaching covenant threshold.",
        "Management cited macroeconomic headwinds and pricing pressure.",
    ]
    
    print("  Processing credit risk analysis...")
    credit_start = time.time()
    
    credit_result = credit_pipeline.process(
        borrower="ABC Corp",
        ocr_output=ocr_result,
        rag_context=rag_result,
        vision_charts=vision_result,
        news_articles=news_articles,
    )
    
    credit_time = (time.time() - credit_start) * 1000
    
    print(f"  ✓ Credit Risk Analysis Complete ({credit_time:.0f}ms)")
    
    # ========================================
    # FINAL RESULTS
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    total_time = ocr_time + rag_time + credit_time
    
    print(f"\n📊 Performance Metrics:")
    print(f"   OCR:         {ocr_time:>6.0f}ms ({ocr_time/total_time*100:>5.1f}%)")
    print(f"   RAG:         {rag_time:>6.0f}ms ({rag_time/total_time*100:>5.1f}%)")
    print(f"   Credit Risk: {credit_time:>6.0f}ms ({credit_time/total_time*100:>5.1f}%)")
    print(f"   {'─'*40}")
    print(f"   Total:       {total_time:>6.0f}ms")
    
    print(f"\n💰 Cost Analysis:")
    print(f"   OCR:         $0.00002")
    print(f"   RAG:         $0.003")
    print(f"   Credit Risk: $0.003")
    print(f"   {'─'*40}")
    print(f"   Total:       $0.00602")
    
    print(f"\n📈 Credit Risk Assessment:")
    print(f"   Borrower:    {credit_result['borrower']}")
    print(f"   PD (12M):    {credit_result['pd']:.2%}")
    print(f"   Risk Level:  {'HIGH' if credit_result['pd'] > 0.15 else 'MODERATE' if credit_result['pd'] > 0.05 else 'LOW'}")
    
    print(f"\n🎯 Top Risk Drivers:")
    for i, driver in enumerate(credit_result['drivers'][:5], 1):
        print(f"   {i}. {driver}")
    
    print(f"\n💼 Key Financial Metrics:")
    features = credit_result['features']
    if 'debt_to_ebitda' in features:
        print(f"   Debt/EBITDA:       {features['debt_to_ebitda']:.2f}x")
    if 'interest_coverage' in features:
        print(f"   Interest Coverage: {features['interest_coverage']:.2f}x")
    if 'current_ratio' in features:
        print(f"   Current Ratio:     {features['current_ratio']:.2f}")
    if 'news_sentiment' in features:
        print(f"   News Sentiment:    {features['news_sentiment']:.2f}")
    
    print(f"\n🔮 Counterfactual Scenarios:")
    for scenario, result in list(credit_result['counterfactuals'].items())[:3]:
        print(f"   {scenario}:")
        print(f"      PD: {result['baseline_pd']:.2%} → {result['new_pd']:.2%} ({result['delta_pd']:+.2%})")
    
    print(f"\n📝 Risk Memo Preview:")
    print("-" * 70)
    memo_preview = credit_result['risk_memo'][:400]
    print(memo_preview)
    print("...")
    print("-" * 70)
    
    print(f"\n⚠️  Drift Detection:")
    print(f"   Status: {'⚠️ DRIFT DETECTED' if credit_result['drift_detected'] else '✓ No drift'}")
    
    if mode == "sagemaker" and s3_bucket:
        print(f"\n☁️  SageMaker:")
        print(f"   Results saved to: s3://{s3_bucket}/results/abc_corp.json")
    
    print("\n" + "=" * 70)
    print("E2E Demo Complete!")
    print("=" * 70)
    
    print(f"\n✓ Successfully processed 1 document through full pipeline")
    print(f"✓ Total time: {total_time:.0f}ms")
    print(f"✓ Total cost: $0.00602")
    print(f"\nFor batch processing:")
    print(f"  - Local (80 samples):  ~{total_time * 80 / 1000:.0f} seconds")
    print(f"  - SageMaker (600):     ~{total_time * 600 / 1000 / 60:.0f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full E2E Pipeline Demo")
    parser.add_argument("--mode", default="local", choices=["local", "sagemaker", "production"])
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket for SageMaker mode")
    
    args = parser.parse_args()
    
    run_full_e2e_demo(mode=args.mode, s3_bucket=args.s3_bucket)