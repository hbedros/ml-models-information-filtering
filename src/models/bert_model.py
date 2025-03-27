# BERT.py (using sentence-transformers)
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer

class BERTEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device or (
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )
        self.model = SentenceTransformer(model_name, device=self.device)

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        return ' '.join([word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])

    def embed_text(self, text):
        preprocessed = self.preprocess(text)
        embedding = self.model.encode(preprocessed, convert_to_numpy=True)
        return embedding

    def embed_dataframe(self, df, text_column, batch_size=32):
        texts = df[text_column].apply(self.preprocess).astype(str).tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return np.array(embeddings)
