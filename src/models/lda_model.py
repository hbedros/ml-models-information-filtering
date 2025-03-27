import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class LDAModel:
    def __init__(self, n_components=2, random_state=42):
        """
        Initialize the LDA model with the specified number of topics.

        Parameters:
        - n_components: int, optional (default=2)
            Number of topics to extract.
        - random_state: int, optional (default=42)
            Random seed for reproducibility.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer = CountVectorizer(stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=self.n_components, random_state=self.random_state)
        self.feature_names = None

    def fit(self, df, text_column):
        """
        Fit the LDA model on the provided DataFrame.

        Parameters:
        - df: pandas.DataFrame
            DataFrame containing the text data.
        - text_column: str
            Name of the column containing text documents.
        """
        # Ensure the text_column exists in the DataFrame
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        # Drop rows where the text_column is NaN
        docs = df[text_column].dropna().tolist()

        # Vectorize the documents
        X = self.vectorizer.fit_transform(docs)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Fit the LDA model
        self.lda.fit(X)

    def print_topics(self, n_top_words=5):
        """
        Print the top words for each topic.

        Parameters:
        - n_top_words: int, optional (default=5)
            Number of top words to display for each topic.
        """
        if self.feature_names is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' with appropriate data before printing topics.")

        for idx, topic in enumerate(self.lda.components_):
            print(f"Topic {idx}: ", " | ".join([self.feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

    def transform(self, df, text_column):
        """
        Transform the documents to topic distributions.

        Parameters:
        - df: pandas.DataFrame
            DataFrame containing the text data.
        - text_column: str
            Name of the column containing text documents.

        Returns:
        - doc_topic_dist: array, shape (n_samples, n_components)
            Document-topic distribution for each document.
        """
        if self.feature_names is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' with appropriate data before transforming documents.")

        # Ensure the text_column exists in the DataFrame
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        # Drop rows where the text_column is NaN
        docs = df[text_column].dropna().tolist()

        # Vectorize the documents
        X = self.vectorizer.transform(docs)

        # Transform the documents to topic distributions
        doc_topic_dist = self.lda.transform(X)
        return doc_topic_dist
