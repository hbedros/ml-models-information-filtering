import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_texts():
    """Fixture providing sample texts for testing."""
    return [
        "This is the first document about machine learning.",
        "The second document is about natural language processing.",
        "Document three discusses deep learning and AI.",
        "The fourth document covers information retrieval.",
        "Fifth document is about text classification and NLP."
    ]

@pytest.fixture
def sample_queries():
    """Fixture providing sample queries for testing."""
    return [
        "machine learning AI",
        "natural language processing",
        "information retrieval"
    ]

@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame for testing."""
    texts = [
        "This is the first document about machine learning.",
        "The second document is about natural language processing.",
        "Document three discusses deep learning and AI.",
        "The fourth document covers information retrieval.",
        "Fifth document is about text classification and NLP."
    ]
    return pd.DataFrame({'text': texts})

@pytest.fixture
def sample_embeddings():
    """Fixture providing sample embeddings for testing."""
    # Create 5 documents with 384-dimensional embeddings (matching MiniLM-L6-v2)
    return np.random.rand(5, 384) 