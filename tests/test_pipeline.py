import pytest
import numpy as np
from src.pipeline.pipeline import (
    preprocess_texts,
    lda_topic_clustering,
    bert_contextual_filtering,
    generate_queries_from_topics,
    pipeline
)

def test_preprocess_texts(sample_texts):
    """Test text preprocessing functionality."""
    processed = preprocess_texts(sample_texts)
    assert len(processed) == len(sample_texts)
    assert all(isinstance(text, str) for text in processed)
    assert all(text.islower() for text in processed)

def test_lda_topic_clustering(sample_texts):
    """Test LDA topic clustering functionality."""
    n_topics = 3
    topic_assignments, lda_model, vectorizer = lda_topic_clustering(sample_texts, n_topics=n_topics)
    
    assert len(topic_assignments) == len(sample_texts)
    assert all(isinstance(topic, (int, np.integer)) for topic in topic_assignments)
    assert all(0 <= topic < n_topics for topic in topic_assignments)
    assert lda_model.n_components == n_topics

def test_bert_contextual_filtering(sample_texts):
    """Test BERT contextual filtering functionality."""
    query = "machine learning"
    relevance, similarities, embeddings = bert_contextual_filtering(
        sample_texts,
        query,
        threshold=0.5
    )
    
    assert len(relevance) == len(sample_texts)
    assert len(similarities) == len(sample_texts)
    assert embeddings.shape[0] == len(sample_texts)
    assert all(isinstance(rel, bool) for rel in relevance)
    assert all(0 <= sim <= 1 for sim in similarities)

def test_generate_queries(sample_texts):
    """Test query generation from topics."""
    n_topics = 3
    _, lda_model, vectorizer = lda_topic_clustering(sample_texts, n_topics=n_topics)
    queries = generate_queries_from_topics(lda_model, vectorizer, top_n=3)
    
    assert len(queries) == n_topics
    assert all(isinstance(query, str) for query in queries)
    assert all(len(query.split()) <= 3 for query in queries)

def test_full_pipeline(sample_dataframe):
    """Test the complete pipeline functionality."""
    n_topics = 3
    processed_df, lda_model, vectorizer = pipeline(
        sample_dataframe,
        text_column='text',
        n_topics=n_topics,
        threshold=0.5
    )
    
    # Check if all expected columns are present
    assert 'topic' in processed_df.columns
    assert any('relevant_to_' in col for col in processed_df.columns)
    assert any('similarity_to_' in col for col in processed_df.columns)
    assert any('embedding_' in col for col in processed_df.columns)
    
    # Check if topic assignments are valid
    assert all(0 <= topic < n_topics for topic in processed_df['topic'])

def test_pipeline_empty_input():
    """Test pipeline behavior with empty input."""
    empty_df = pd.DataFrame({'text': []})
    with pytest.raises(ValueError, match="No valid texts found in the DataFrame"):
        pipeline(empty_df, text_column='text')

def test_pipeline_invalid_input():
    """Test pipeline behavior with invalid input."""
    invalid_df = pd.DataFrame({'text': [None, None]})
    with pytest.raises(ValueError, match="No valid texts found in the DataFrame"):
        pipeline(invalid_df, text_column='text') 