import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

class NaiveBayesBigram:
    def __init__(self):
        # Initialize the CountVectorizer with bigram features
        self.vectorizer = CountVectorizer(ngram_range=(2, 2))  # Bigram feature extraction
        self.model = MultinomialNB()

    def train(self, df, text_column, label_column):
        """
        Train the Naive Bayes model on bigram features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): The column containing text data.
            label_column (str): The column containing labels.

        Returns:
            dict: A dictionary containing training metrics.
        """
        # Extract features and labels
        X = self.vectorizer.fit_transform(df[text_column])
        y = df[label_column]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Naive Bayes model
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return {"accuracy": accuracy, "classification_report": report}

    def predict(self, texts):
        """
        Predict labels for new text data.

        Args:
            texts (list[str]): A list of text strings.

        Returns:
            list: Predicted labels.
        """
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)