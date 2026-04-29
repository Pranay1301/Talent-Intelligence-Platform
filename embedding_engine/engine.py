"""
Embedding Engine - Sentence-BERT based semantic embedding and similarity.
Generates contextual embeddings for semantic matching between resumes and JDs.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Hybrid embedding engine combining Sentence-BERT dense embeddings
    with TF-IDF sparse representations for robust semantic matching.
    
    This allows "Machine Learning Engineer" to match with
    "Built predictive AI systems" even without exact keyword overlap.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with Sentence-BERT model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._tfidf_fitted = False
        logger.info("Embedding engine initialized successfully")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate dense embeddings for a list of texts."""
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            texts, batch_size=batch_size,
            show_progress_bar=False, normalize_embeddings=True,
        )
        return np.array(embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.encode([text])[0]

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        return float(cosine_similarity(embedding1, embedding2)[0][0])

    def compute_similarity_matrix(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        return cosine_similarity(embeddings1, embeddings2)

    def compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Compute TF-IDF based cosine similarity between two texts."""
        try:
            tfidf_matrix = self.tfidf.fit_transform([text1, text2])
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception as e:
            logger.warning(f"TF-IDF similarity failed: {e}")
            return 0.0

    def hybrid_similarity(self, text1: str, text2: str,
                          dense_weight: float = 0.7,
                          sparse_weight: float = 0.3) -> dict:
        """
        Compute hybrid similarity combining dense (SBERT) and sparse (TF-IDF) scores.
        
        Returns:
            Dictionary with individual and combined scores.
        """
        # Dense similarity (Sentence-BERT)
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        dense_score = self.compute_similarity(emb1, emb2)

        # Sparse similarity (TF-IDF)
        sparse_score = self.compute_tfidf_similarity(text1, text2)

        # Weighted combination
        combined = dense_weight * dense_score + sparse_weight * sparse_score

        return {
            "dense_score": round(dense_score, 4),
            "sparse_score": round(sparse_score, 4),
            "combined_score": round(combined, 4),
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
        }

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()
