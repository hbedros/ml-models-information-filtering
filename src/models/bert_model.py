# BERT.py

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class BERTEmbedder:
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        return ' '.join([word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])

    def embed_text(self, text):
        preprocessed = self.preprocess(text)
        inputs = self.tokenizer(preprocessed, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_embedding.cpu().numpy().flatten()

    def embed_dataframe(self, df, text_column):
        return np.array([self.embed_text(row) for row in df[text_column]])
