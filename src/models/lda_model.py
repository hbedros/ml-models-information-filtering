import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

class LDAModel:
    def __init__(self, num_topics=5, passes=10):
        self.num_topics = num_topics
        self.passes = passes
        self.stop_words = set(stopwords.words('english'))
        self.dictionary = None
        self.corpus = None
        self.model = None

    def preprocess(self, documents):
        return [
            [word for word in word_tokenize(doc.lower()) if word.isalpha() and word not in self.stop_words]
            for doc in documents
        ]

    def train(self, raw_documents):
        texts = self.preprocess(raw_documents)
        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

        self.model = gensim.models.LdaModel(
            self.corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            passes=self.passes,
            random_state=42
        )

    def get_topics(self, num_words=5):
        if self.model is None:
            raise ValueError("LDA model not trained yet.")
        return self.model.print_topics(num_words=num_words)

    def infer_topic_distribution(self, new_doc):
        bow = self.dictionary.doc2bow(self.preprocess([new_doc])[0])
        return self.model.get_document_topics(bow)
