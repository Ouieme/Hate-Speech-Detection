import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Ensure stopwords are available
import nltk
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocesses the input text by removing special characters,
    converting to lowercase, and removing stopwords.
    """
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\\s]', '', text)
    # Lowercase the text
    text = text.lower()
    # Tokenize and remove stopwords
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def load_glove_embeddings(file_path):
    """
    Loads GloVe embeddings from a file.
    """
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

def get_sentence_embedding(sentence, embeddings_index, embedding_dim=100):
    """
    Creates an embedding for the sentence using the GloVe embeddings.
    """
    words = sentence.split()
    valid_embeddings = [embeddings_index.get(word, np.zeros(embedding_dim)) for word in words]
    if valid_embeddings:
        return np.mean(valid_embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)
