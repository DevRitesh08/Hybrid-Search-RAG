"""Chunking strategies and factory interface for Hybrid Search RAG."""

from hybrid_rag.chunking.base import BaseChunker, ChunkerConfig, get_chunker
from hybrid_rag.chunking.fixed import FixedChunker
from hybrid_rag.chunking.fixed import chunk_document as chunk_document_fixed
from hybrid_rag.chunking.semantic import SemanticChunker
from hybrid_rag.chunking.semantic import chunk_document as chunk_document_semantic
from hybrid_rag.chunking.structural import StructuralChunker
from hybrid_rag.chunking.structural import chunk_document as chunk_document_structural

__all__ = [
    "BaseChunker",
    "ChunkerConfig",
    "FixedChunker",
    "SemanticChunker",
    "StructuralChunker",
    "chunk_document_fixed",
    "chunk_document_semantic",
    "chunk_document_structural",
    "get_chunker",
]
