"""
RAG System Demo

Demonstrates the RAG pipeline:
1. Document chunking
2. Hybrid retrieval (BM25 + Dense)
3. Cross-encoder reranking
4. Answer generation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from rag_system.chunking import DocumentChunker
from rag_system.retrieval import HybridRetriever
from rag_system.reranking import BGEReranker


def main():
    """Run RAG demo"""
    print("=" * 60)
    print("RAG System Demo")
    print("=" * 60)
    
    # Sample financial documents
    documents = [
        """
# Q3 2025 Financial Report

## Revenue Summary
Total revenue for Q3 2025 was $2.5M, representing a 19% increase from Q2's $2.1M.

Product sales contributed $1.8M (72% of revenue), while services generated $0.7M (28%).

## Key Metrics
- Customer acquisition: 1,200 new customers
- Average contract value: $12,500
- Churn rate: 3.2% (down from 4.1% in Q2)
- Customer lifetime value: $45,000
        """,
        """
# Annual Growth Targets 2025

The company has set ambitious growth targets for 2025:
- Revenue growth: 15% year-over-year
- Customer base expansion: 5,000 new customers
- Market share: Increase from 12% to 18%

These targets were approved by the board on January 15, 2025.
        """,
        """
# Customer Acquisition Analysis

Recent marketing campaigns have reduced customer acquisition cost (CAC) to $450 per customer, 
down from $620 in the previous quarter.

The improvement is attributed to:
1. More targeted digital advertising
2. Improved conversion funnel
3. Better qualified leads from partnerships
        """,
    ]
    
    # Step 1: Document Chunking
    print("\n" + "-" * 60)
    print("Step 1: Document Chunking")
    print("-" * 60)
    
    chunker = DocumentChunker(
        chunk_size=512,
        chunk_overlap=128,
        preserve_tables=True,
        preserve_lists=True
    )
    
    all_chunks = []
    for i, doc in enumerate(documents):
        chunks = chunker.chunk_document(
            doc,
            metadata={"doc_id": i, "source": f"document_{i}.md"}
        )
        all_chunks.extend(chunks)
        print(f"Document {i}: {len(chunks)} chunks created")
    
    stats = chunker.get_chunk_stats(all_chunks)
    print(f"\nChunking Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Chunk types: {stats['chunk_types']}")
    print(f"  Average size: {stats['avg_chunk_size']:.0f} characters")
    
    # Step 2: Build Retrieval Index
    print("\n" + "-" * 60)
    print("Step 2: Building Retrieval Index")
    print("-" * 60)
    
    retriever = HybridRetriever(
        embedding_model="BAAI/bge-m3",
        top_k_final=5
    )
    
    print("Building BM25 and FAISS indices...")
    retriever.build_index(all_chunks)
    print("✓ Indices built successfully")
    
    # Step 3: Test Retrieval
    print("\n" + "-" * 60)
    print("Step 3: Hybrid Retrieval")
    print("-" * 60)
    
    test_queries = [
        "What was our Q3 revenue?",
        "How does Q3 compare to Q2?",
        "What is our customer acquisition cost?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        results = retriever.retrieve(query)
        
        print(f"Retrieved {len(results)} chunks:")
        for i, (chunk, score) in enumerate(results[:3]):  # Show top 3
            print(f"\n{i+1}. Score: {score:.4f}")
            print(f"   Type: {chunk.metadata.get('chunk_type', 'unknown')}")
            print(f"   Text: {chunk.text[:150]}...")
    
    # Step 4: Reranking
    print("\n" + "-" * 60)
    print("Step 4: Cross-Encoder Reranking")
    print("-" * 60)
    
    reranker = BGEReranker()
    
    query = test_queries[0]
    print(f"Query: '{query}'")
    
    # Get initial retrieval results
    initial_results = retriever.retrieve(query)
    print(f"\nBefore reranking (top 3):")
    for i, (chunk, score) in enumerate(initial_results[:3]):
        print(f"  {i+1}. Score: {score:.4f} | {chunk.text[:80]}...")
    
    # Rerank
    reranked_results = reranker.rerank(query, initial_results, top_k=5)
    print(f"\nAfter reranking (top 3):")
    for i, (chunk, score) in enumerate(reranked_results[:3]):
        print(f"  {i+1}. Score: {score:.4f} | {chunk.text[:80]}...")
    
    print("\n" + "=" * 60)
    print("RAG Demo Complete!")
    print("=" * 60)
    print("\nNext: Run 03_e2e_demo.py to see the complete OCR→RAG pipeline")


if __name__ == "__main__":
    main()