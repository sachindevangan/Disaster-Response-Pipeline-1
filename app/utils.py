import nltk
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
import re

url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

def tokenize(text):
    """ 
    Tokenize Function. 
  
    Cleaning The Data And Tokenizing Text. 
  
    Parameters: 
    text (str): Text For Cleaning And Tokenizing (English).
    
    Returns: 
    clean_tokens (List): Tokenized Text, Clean For ML Modeling
    """

    # removing urls 
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenizing
    tokens = word_tokenize(text)
    
    # lemmatizing
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
