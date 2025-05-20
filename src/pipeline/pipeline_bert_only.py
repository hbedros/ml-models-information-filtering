import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from models.bert_model import BERTEmbedder

# Step 1: Preprocessing
def preprocess_texts(texts):
    return [text.lower() for text in texts if isinstance(text, str)]

# BERT Contextual Filtering
def bert_contextual_filtering(texts, query, model_name="all-MiniLM-L6-v2", threshold=0.5):
    embedder = BERTEmbedder(model_name=model_name)
    query_embedding = embedder.embed_text(query)
    df = pd.DataFrame({'text': texts})
    text_embeddings = embedder.embed_dataframe(df, 'text')

    similarities = np.dot(text_embeddings, query_embedding) / (
        np.linalg.norm(text_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    relevance = similarities > threshold
    return relevance, similarities, text_embeddings

# Main pipeline — **No LDA**
def pipeline(dataframe, text_column, queries, bert_model="all-MiniLM-L6-v2", threshold=0.5):
    texts = preprocess_texts(dataframe[text_column])

    results = {}
    all_embeddings = None
    combined_relevance = np.zeros(len(texts), dtype=bool)

    for i, query in enumerate(queries):
        relevance, similarities, text_embeddings = bert_contextual_filtering(
            texts, query, bert_model, threshold
        )

        if all_embeddings is None:
            all_embeddings = text_embeddings

        combined_relevance |= relevance

        results[f'query_{i}'] = {
            'relevance': relevance,
            'similarities': similarities
        }

    filtered_texts = [t for t, keep in zip(texts, combined_relevance) if keep]
    filtered_embeddings = all_embeddings[combined_relevance]

    return {
        'query_results': results,
        'embeddings': filtered_embeddings,
        'filtered_texts': filtered_texts,
        'combined_relevance': combined_relevance
    }

# PDF processing wrapper — **No LDA**
def process_pdf_with_pipeline(pdf_file_path: str, queries, threshold=0.5, bert_model="all-MiniLM-L6-v2"):
    reader = PdfReader(pdf_file_path)
    texts = [page.extract_text() for page in reader.pages]

    df = pd.DataFrame({
        'text': texts,
        'page': range(1, len(texts) + 1)
    })

    results = pipeline(
        df, 'text',
        queries=queries,
        threshold=threshold,
        bert_model=bert_model
    )

    results_df = pd.DataFrame({'text': results['filtered_texts']})

    for query_id, result in results['query_results'].items():
        relevant_similarities = result['similarities'][results['combined_relevance']]
        relevant_flags = result['relevance'][results['combined_relevance']]

        results_df[f'{query_id}_similarity'] = relevant_similarities
        results_df[f'{query_id}_relevant'] = relevant_flags

    embeddings = results['embeddings']
    for dim in range(embeddings.shape[1]):
        results_df[f'embedding_{dim}'] = embeddings[:, dim]

    return results_df
