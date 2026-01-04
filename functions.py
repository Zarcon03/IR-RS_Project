import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

from rake_nltk import Rake
from keybert import KeyBERT

from nltk.wsd import lesk
from nltk import word_tokenize

def keywords_extraction_RAKE(text: str, n: int) -> list[str]:
    r = Rake()
    r.extract_keywords_from_text(text)
    phrases = r.get_ranked_phrases()[:n]
    words = set()
    for phrase in phrases:
        for word in phrase.split():
            words.add(word.lower())
    return list(words)

def keywords_extraction_BERT(text: str, n: int) -> list[str]:
    kwBert = KeyBERT()
    keywords = kwBert.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=n)
    kw = []
    for k in keywords:
        kw.append(k[0])
    return kw

def keywords_extractor(text: str, n=3, method='rake') -> list[str]:
    if method not in ['rake', 'bert']:
        raise ValueError("Method must be either 'rake' or 'bert'")
    
    if method == 'rake':
        keywords = keywords_extraction_RAKE(text, n)
        return keywords
    
    if method == 'bert':
        keywords = keywords_extraction_BERT(text, n)
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