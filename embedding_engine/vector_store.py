"""
FAISS Vector Store - Production-grade vector search for candidate retrieval.
Enables fast nearest-neighbor search across large resume datasets.
"""

import os
import json
import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    
    Supports:
    - Adding embeddings with metadata
    - Fast nearest-neighbor search
    - Index persistence (save/load)
    - Batch operations for large datasets
    """

    def __init__(self, dimension: int = 384, index_path: str = None, metadata_path: str = None):
        """Initialize FAISS index."""
        self.dimension = dimension
        self.index_path = index_path or "./faiss_index/index.faiss"
        self.metadata_path = metadata_path or "./faiss_index/metadata.json"
        self.metadata: list[dict] = []

        # Try to load existing index
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load()
        else:
            # Create new index (IndexFlatIP for cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
            logger.info(f"Created new FAISS index with dimension {dimension}")

    def add(self, embeddings: np.ndarray, metadata_list: list[dict]) -> list[int]:
        """
        Add embeddings with associated metadata to the index.
        
        Args:
            embeddings: numpy array of shape (n, dimension)
            metadata_list: list of metadata dicts for each embedding
            
        Returns:
            List of assigned IDs
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        start_id = self.index.ntotal
        self.index.add(embeddings.astype(np.float32))
        
        ids = list(range(start_id, start_id + len(metadata_list)))
        for i, meta in enumerate(metadata_list):
            meta["faiss_id"] = ids[i]
            self.metadata.append(meta)

        logger.info(f"Added {len(ids)} vectors. Total: {self.index.ntotal}")
        return ids

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """
        Search for most similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of dicts with metadata and similarity scores
        """
        if self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)
        query_embedding = query_embedding.astype(np.float32)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata) and idx >= 0:
                result = self.metadata[idx].copy()
                result["similarity_score"] = round(float(score), 4)
                results.append(result)

        return results

    def remove(self, ids: list[int]) -> None:
        """Remove vectors by their IDs (rebuilds index)."""
        if not ids:
            return
        ids_set = set(ids)
        # Collect remaining vectors and metadata
        remaining_meta = []
        remaining_vectors = []
        for i, meta in enumerate(self.metadata):
            if meta.get("faiss_id") not in ids_set:
                remaining_meta.append(meta)
                # Reconstruct vector from index
                vec = self.index.reconstruct(i)
                remaining_vectors.append(vec)

        # Rebuild index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        if remaining_vectors:
            embeddings = np.array(remaining_vectors)
            self.add(embeddings, remaining_meta)

    def save(self) -> None:
        """Persist index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors)")

    def load(self) -> None:
        """Load index and metadata from disk."""
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors)")

    def clear(self) -> None:
        """Clear the entire index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info("FAISS index cleared")

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal
