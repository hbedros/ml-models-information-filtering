import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer, util
import torch

# Step 1: Preprocessing
def preprocess_texts(texts):
    # Basic preprocessing: remove NaN, lowercase, etc.
    return [text.lower() for text in texts if isinstance(text, str)]

# Step 2: LDA Topic Clustering
def lda_topic_clustering(texts, n_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    topic_assignments = lda.transform(X).argmax(axis=1)  # Assign each document to its most probable topic
    return topic_assignments, lda, vectorizer

# Step 3: Sentence-Transformers for Contextual Filtering
def bert_contextual_filtering(texts, query, model_name="all-MiniLM-L6-v2", threshold=0.5):
    """
    Use Sentence-Transformers to compute embeddings and filter texts based on similarity to a query.
    """
    # Load the model and move it to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name).to(device)
    
    # Compute embeddings for the query and texts
    query_embedding = model.encode(query, convert_to_tensor=True, device=device)
    text_embeddings = model.encode(texts, convert_to_tensor=True, device=device)
    
    # Compute cosine similarity
    similarities = util.cos_sim(query_embedding, text_embeddings)[0]
    
    # Filter texts based on the similarity threshold
    results = [similarity.item() > threshold for similarity in similarities]
    return results, similarities.cpu().numpy()

# Step 4: Pipeline Integration
def pipeline(dataframe, text_column, query, n_topics=5, bert_model="all-MiniLM-L6-v2", threshold=0.5):
    """
    Full pipeline: LDA for topic clustering + Sentence-Transformers for contextual filtering.
    """
    # Preprocess texts
    texts = preprocess_texts(dataframe[text_column])
    
    # LDA Topic Clustering
    topic_assignments, lda_model, vectorizer = lda_topic_clustering(texts, n_topics=n_topics)
    dataframe['topic'] = topic_assignments
    
    # Sentence-Transformers Contextual Filtering
    relevance, similarities = bert_contextual_filtering(texts, query, model_name=bert_model, threshold=threshold)
    dataframe['relevant'] = relevance
    dataframe['similarity'] = similarities  # Store similarity scores for analysis
    
    return dataframe, lda_model, vectorizer

# Example Usage
if __name__ == "__main__":
    # Load data (e.g., true_df)
    from utils.utils import true_df  # Corrected import for utils module
    
    # Define a query for contextual filtering
    query = "climate change and global warming"
    
    # Run the pipeline
    processed_df, lda_model, vectorizer = pipeline(true_df, text_column='content', query=query, n_topics=5)
    
    # Preview results
    print(processed_df.head())
