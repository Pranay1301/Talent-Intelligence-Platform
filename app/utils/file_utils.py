"""
File handling utilities for resume uploads.
"""

import os
import hashlib
from datetime import datetime


def validate_file(filename: str, allowed_extensions: list[str], max_size_mb: int = 10) -> bool:
    """Validate uploaded file by extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions


def generate_unique_filename(original_name: str, prefix: str = "") -> str:
    """Generate unique filename with timestamp hash."""
    name, ext = os.path.splitext(original_name)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    hash_suffix = hashlib.md5(f"{name}{timestamp}".encode()).hexdigest()[:6]
    return f"{prefix}{timestamp}_{hash_suffix}{ext}"


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


def ensure_directory(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
