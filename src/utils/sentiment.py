from textblob import TextBlob
import pandas as pd

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text using TextBlob.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing polarity and subjectivity scores.
    """
    if not isinstance(text, str) or not text.strip():
        return {"polarity": None, "subjectivity": None}
    
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}

def add_sentiment_to_dataframe(df, text_column='text'):
    """
    Add sentiment analysis results (polarity and subjectivity) to a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing text data.

    Returns:
        pd.DataFrame: The DataFrame with added 'polarity' and 'subjectivity' columns.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the DataFrame.")
    
    # Apply sentiment analysis to each row in the text column
    sentiments = df[text_column].dropna().apply(analyze_sentiment)
    sentiment_df = pd.DataFrame(list(sentiments))
    
    # Add polarity and subjectivity columns to the original DataFrame
    df = pd.concat([df, sentiment_df], axis=1)
    return df