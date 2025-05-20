import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from models.bert_model import BERTEmbedder
from models.lda_model import LDAModel
import re

# Download NLTK sentence tokenizer if needed
nltk.download('punkt')

def preprocess_texts(texts):
    return [text.lower() for text in texts if isinstance(text, str)]

# def split_into_sentences(text):
#    return sent_tokenize(text)

def split_into_sentences(text):
    # Basic sentence splitter using punctuation
    return re.split(r'(?<=[.!?])\s+', text.strip())


def lda_topic_clustering(texts, n_topics=5):
    df = pd.DataFrame({'text': texts})
    lda_model = LDAModel(n_components=n_topics, random_state=42)
    lda_model.fit(df, 'text')
    topic_distributions = lda_model.transform(df, 'text')
    topic_assignments = topic_distributions.argmax(axis=1)
    return topic_assignments, lda_model

def bert_contextual_filtering(sentences, query, model_name="all-MiniLM-L6-v2", threshold=0.3):
    embedder = BERTEmbedder(model_name=model_name)
    query_embedding = embedder.embed_text(query)
    df = pd.DataFrame({'text': sentences})
    sentence_embeddings = embedder.embed_dataframe(df, 'text')

    similarities = np.dot(sentence_embeddings, query_embedding) / (
        np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    relevance = similarities > threshold
    return relevance, similarities, sentence_embeddings

def generate_queries_from_topics(lda_model, top_n=5):
    topics = []
    for topic_idx in range(lda_model.n_components):
        top_words = lda_model.print_topics(n_top_words=top_n)
        topics.append(top_words)
    return topics

def pipeline(dataframe, text_column, queries=None, n_topics=5, bert_model="all-MiniLM-L6-v2", threshold=0.3, top_n_words=5):
    texts = preprocess_texts(dataframe[text_column])
    topic_assignments, lda_model = lda_topic_clustering(texts, n_topics)

    if queries is None:
        queries = generate_queries_from_topics(lda_model, top_n_words)

    results = {}
    all_filtered_sentences = []
    all_embeddings = []

    for text in texts:
        sentences = split_into_sentences(text)

        # Aggregate relevance across all queries
        combined_relevance = np.zeros(len(sentences), dtype=bool)

        for i, query in enumerate(queries):
            relevance, similarities, sentence_embeddings = bert_contextual_filtering(
                sentences, query, bert_model, threshold
            )
            combined_relevance |= relevance

            results.setdefault(f'query_{i}', {'relevance': [], 'similarities': []})
            results[f'query_{i}']['relevance'].append(relevance)
            results[f'query_{i}']['similarities'].append(similarities)

        # Filter sentences
        filtered_sentences = [s for s, keep in zip(sentences, combined_relevance) if keep]
        all_filtered_sentences.append(" ".join(filtered_sentences))

        # Optional: average embeddings of kept sentences
        if combined_relevance.any():
            filtered_embeddings = sentence_embeddings[combined_relevance]
            mean_embedding = np.mean(filtered_embeddings, axis=0)
        else:
            mean_embedding = np.zeros(sentence_embeddings.shape[1])  # Blank embedding if nothing matched

        all_embeddings.append(mean_embedding)

    all_embeddings = np.vstack(all_embeddings)

    return {
        'filtered_texts': all_filtered_sentences,
        'embeddings': all_embeddings,
        'query_results': results,
        'topic_assignments': topic_assignments,
        'lda_model': lda_model
    }
