from rake_nltk import Rake
from keybert import KeyBERT
import nltk
from nltk.wsd import lesk
from nltk import word_tokenize
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from constants import DEVICE

# 1. DEFINE GLOBAL REFERENCE
_KW_MODEL_BERT: Optional[KeyBERT] = None
_KW_MODEL_Rake: Optional[Rake] = None

def _initialize_model(model):
    # Internal helper to load the model only when requested.
    if model == "BERT":
        print(f"Initializing KeyBERT on: {DEVICE}...")
        return KeyBERT()
    elif model == "Rake":
        return Rake()


def keywords_extraction_BERT(text: str, max_keywords: int) -> List[str]:
    # Extracts keywords using KeyBERT. Initializes the model only on the first run (Lazy Loading).
    global _KW_MODEL_BERT
    if _KW_MODEL_BERT is None:
        _KW_MODEL_BERT = _initialize_model("BERT")

    keywords = _KW_MODEL_BERT.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 1), 
        stop_words='english', 
        top_n=max_keywords)
    
    return [k[0] for k in keywords]

def keywords_extraction_RAKE(text: str, max_keywords: int) -> list[str]:
    global _KW_MODEL_Rake
    if _KW_MODEL_Rake is None:
        _KW_MODEL_Rake = _initialize_model("Rake")

    r = _KW_MODEL_Rake
    r.extract_keywords_from_text(text)
    phrases = r.get_ranked_phrases()
    
    words = set()
    for phrase in phrases:
        for word in phrase.split():
            if len(words) >= max_keywords:
                return list(words)
            words.add(word.lower())
    
    return list(words)

def keywords_extractor(text: str, max_keywords=3, method='rake') -> list[str]:
    if method not in ['rake', 'bert']:
        raise ValueError("Method must be either 'rake' or 'bert'")
    
    if method == 'rake':
        keywords = keywords_extraction_RAKE(text, max_keywords)
        return keywords
    
    if method == 'bert':
        keywords = keywords_extraction_BERT(text, max_keywords)
        return keywords

def thesaurus_based_expansion(text: str, keywords: list[str], max_synonyms_per_keyword=2) -> list[str]:
    expanded_keywords = []
    for kw in keywords:
        synset = lesk(word_tokenize(text), kw)

        if synset:
            synonyms = set()
            for lemma in synset.lemmas():
                if lemma.name().lower() != kw.lower():
                    synonyms.add(lemma.name().replace('_', ' '))
                if len(synonyms) >= max_synonyms_per_keyword:
                    break

            expanded_keywords.append(kw)
            expanded_keywords.extend(synonyms)

        else:
            expanded_keywords.append(kw)
    
    return expanded_keywords

class TokenizerWrapper:
    # Tokenizer that actually works with py.sliding
    def __init__(self, tok):
        self.tok = tok

    def tokenize(self, s: str):
        return self.tok.tokenize(s)

    def convert_tokens_to_string(self, tokens):
        # PyTerrier may pass a tuple;
        # T5Tokenizer expects something mutable
        return self.tok.convert_tokens_to_string(list(tokens))
    
def _download_nltk_resources():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/stopwords', 'stopwords')
    ]
    for resource, download_name in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(download_name)
            