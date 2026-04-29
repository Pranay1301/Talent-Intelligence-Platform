"""
Text preprocessing utilities for NLP pipeline.
"""

import re
import string


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep useful punctuation
    text = re.sub(r"[^\w\s.,;:!?@#$%&*()\-+=/']", "", text)
    return text.strip()


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def normalize_skill(skill: str) -> str:
    """Normalize a skill name for comparison."""
    skill = skill.lower().strip()
    # Common normalizations
    mappings = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "ml": "machine learning",
        "dl": "deep learning",
        "tf": "tensorflow",
        "k8s": "kubernetes",
        "postgres": "postgresql",
        "mongo": "mongodb",
    }
    return mappings.get(skill, skill)


def compute_text_overlap(text1: str, text2: str) -> float:
    """Compute word-level overlap between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    # Remove stopwords
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                 "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "could",
                 "should", "may", "might", "can", "shall", "this", "that", "these",
                 "those", "i", "we", "you", "he", "she", "it", "they"}
    words1 -= stopwords
    words2 -= stopwords
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)
