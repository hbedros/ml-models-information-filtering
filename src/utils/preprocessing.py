import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Initialize stop words, stemmer, and lemmatizer
STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

def clean_text(text, remove_stopwords=True, use_stemming=False, use_lemmatization=False):
    """
    Clean text by removing special characters, stop words, and applying stemming or lemmatization.

    Args:
        text (str): Input text to clean.
        remove_stopwords (bool): Whether to remove stop words.
        use_stemming (bool): Whether to apply stemming.
        use_lemmatization (bool): Whether to apply lemmatization.

    Returns:
        str: Cleaned text.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize text
    words = text.split()
    
    # Remove stop words
    if remove_stopwords:
        words = [word for word in words if word not in STOP_WORDS]
    
    # Apply stemming
    if use_stemming:
        words = [STEMMER.stem(word) for word in words]
    
    # Apply lemmatization
    if use_lemmatization:
        words = [LEMMATIZER.lemmatize(word) for word in words]
    
    # Rejoin words into a single string
    return ' '.join(words)
