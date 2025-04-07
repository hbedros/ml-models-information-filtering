import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer, util
import torch
from PyPDF2 import PdfReader
import numpy as np

# Step 1: Preprocessing
def preprocess_texts(texts):
    # Basic preprocessing: remove NaN, lowercase, etc.
    return [text.lower() for text in texts if isinstance(text, str)]

# LDA Topic Clustering
def lda_topic_clustering(texts, n_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    topic_assignments = lda.transform(X).argmax(axis=1)  # Assign each document to its most probable topic
    return topic_assignments, lda, vectorizer

# Sentence-Transformers for Contextual Filtering
def bert_contextual_filtering(texts, query, model_name="all-MiniLM-L6-v2", threshold=0.5):
    """
    Use Sentence-Transformers to compute embeddings, relevance, and similarity to a query.
    """
    # Load the model and move it to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name).to(device)
    
    # Compute embeddings for the query and texts
    query_embedding = model.encode(query, convert_to_tensor=True, device=device)
    text_embeddings = model.encode(texts, batch_size=128, convert_to_tensor=True, device=device)
    
    # Another way to compute text embeddings in batches
    # text_embeddings = [] s
    # for i in range(0, len(texts), batch_size):
    #    batch = texts[i:i + batch_size]
    #    batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
    #    text_embeddings.append(batch_embeddings)
    # text_embeddings = torch.cat(text_embeddings, dim=0)

    # Compute cosine similarity
    similarities = util.cos_sim(query_embedding, text_embeddings)[0]
    
    # Filter texts based on the similarity threshold
    relevance = [similarity.item() > threshold for similarity in similarities]
    
    # Debugging outputs
    print("Query Embedding Shape:", query_embedding.shape)
    print("Text Embeddings Shape:", text_embeddings.shape)
    print("Similarities Shape:", similarities.shape)

    # Return relevance, similarity scores, and text embeddings
    return relevance, similarities.cpu().numpy(), text_embeddings.cpu().numpy()

# Function to Generate Queries
def generate_queries_from_topics(lda_model, vectorizer, top_n=5):
    """
    Generate queries based on the top words for each LDA topic.
    
    Args:
        lda_model: Trained LDA model.
        vectorizer: Fitted CountVectorizer used for LDA.
        top_n: Number of top words to extract for each topic.
    
    Returns:
        List of queries, one for each topic.
    """
    queries = []
    for topic_idx, topic in enumerate(lda_model.components_):
        # Get the top words for the topic
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-top_n - 1:-1]]
        # Combine the top words into a single query string
        queries.append(" ".join(top_words))
    return queries

# Pipeline Integration
def pipeline(dataframe, text_column, queries=None, n_topics=5, bert_model="all-MiniLM-L6-v2", threshold=0.5, top_n_words=5):
    """
    Hybrid LDA + BERT pipeline for filtering qualitative data.
    """
    # Preprocess the texts
    texts = preprocess_texts(dataframe[text_column])
    
    if not texts:
        raise ValueError("No valid texts found in the DataFrame.")
    
    # LDA Topic Clustering
    topic_assignments, lda_model, vectorizer = lda_topic_clustering(texts, n_topics=n_topics)
    dataframe['topic'] = topic_assignments
    
    # Generate queries from LDA topics if no queries are provided
    if queries is None:
        queries = generate_queries_from_topics(lda_model, vectorizer, top_n=top_n_words)
        print(f"Generated Queries from Topics: {queries}")
    
    if not queries:
        raise ValueError("No queries were generated or provided.")
    
    # Sentence-Transformers Contextual Filtering and Embedding Generation
    all_embeddings = []
    for query in queries:
        relevance, similarities, embeddings = bert_contextual_filtering(texts, query, model_name=bert_model, threshold=threshold)
        dataframe[f'relevant_to_{query}'] = relevance
        dataframe[f'similarity_to_{query}'] = similarities
        all_embeddings.append(embeddings)
    
    if not all_embeddings:
        raise ValueError("No embeddings were generated. Check the queries or texts.")
    
    # Combine all embeddings into a single array
    combined_embeddings = np.mean(all_embeddings, axis=0)  # Average embeddings across queries
    embedding_columns = [f"embedding_{i}" for i in range(combined_embeddings.shape[1])]
    for i, col in enumerate(embedding_columns):
        dataframe[col] = combined_embeddings[:, i]
    
    return dataframe, lda_model, vectorizer

# Process PDF with the pipeline
def process_pdf_with_pipeline(pdf_file_path: str, n_topics=5, threshold=0.5, bert_model="all-MiniLM-L6-v2", top_n_words=5):
    """
    Process a PDF file using the LDA + BERT pipeline.
    
    Args:
        pdf_file_path (str): Path to the PDF file.
        n_topics (int): Number of topics for LDA.
        threshold (float): Similarity threshold for BERT filtering.
        bert_model (str): Pre-trained Sentence-Transformers model name.
        top_n_words (int): Number of top words to extract for query generation.
    
    Returns:
        pd.DataFrame: Processed DataFrame with filtered and clustered text.
    """
    from PyPDF2 import PdfReader
    import pandas as pd

    # Step 1: Extract text from the PDF
    reader = PdfReader(pdf_file_path)
    pdf_texts = [page.extract_text() for page in reader.pages if page.extract_text()]  # Extract text from each page
    
    # Debug: Print sample extracted text
    print("=== Sample Extracted Text ===")
    print(pdf_texts[:5])
    
    # Preprocess the extracted text
    pdf_texts = [text.strip() for text in pdf_texts if text.strip()]  # Remove blank lines and strip whitespace
    
    # Convert the extracted text into a DataFrame
    pdf_df = pd.DataFrame(pdf_texts, columns=["text"])
    pdf_df = pdf_df.dropna().reset_index(drop=True)  # Remove empty rows
    
    # Step 2: Process the extracted text using the pipeline
    processed_pdf_df, lda_model, vectorizer = pipeline(
        pdf_df,
        text_column="text",
        queries=None,  # Let the pipeline generate queries automatically
        n_topics=n_topics,
        bert_model=bert_model,
        threshold=threshold,
        top_n_words=top_n_words
    )
    
    # Debug: Print generated queries
    print("Generated Queries:", [col for col in processed_pdf_df.columns if "relevant_to" in col])
    
    # Step 3: Filter out irrelevant rows
    relevant_rows = processed_pdf_df[[col for col in processed_pdf_df.columns if "relevant_to" in col]].any(axis=1)
    processed_pdf_df = processed_pdf_df[relevant_rows]
    
    # Debug: Print relevance counts
    relevance_columns = [col for col in processed_pdf_df.columns if "relevant_to" in col]
    print("Relevance Counts:")
    for col in relevance_columns:
        print(f"{col}: {processed_pdf_df[col].value_counts()}")
    
    # Debug: Print the shape of the processed DataFrame
    print("Processed DataFrame Shape:", processed_pdf_df.shape)
    
    return processed_pdf_df
