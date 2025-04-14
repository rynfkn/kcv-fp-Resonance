import re
import torch
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import AutoTokenizer, AutoModel
nltk.download('punkt')
nltk.download('stopwords')

class TextProcessor:
    """Processes text data and extracts embeddings"""
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Sastrawi stemmer
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
    
    def preprocess_text(self, text):
        """Apply text preprocessing"""
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        indonesian_stopwords = set(stopwords.words('indonesian'))
        tokens = [word for word in tokens if word not in indonesian_stopwords]
        
        # Apply stemming
        tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    
    def process(self, texts):
        """Extract text embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Add E5 prefix
        batch = ["passage: " + text for text in processed_texts]
        
        # Tokenize and get embeddings
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # E5 uses CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create feature names and dataframe
        columns = [f'e5_{i}' for i in range(embeddings.shape[1])]
        return pd.DataFrame(embeddings, columns=columns)