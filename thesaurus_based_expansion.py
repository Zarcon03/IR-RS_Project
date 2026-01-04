from keywords_extraction import keywords_extractor
import time

text = 'Who gave Hamilton the substance of what he had proposed on the part of General Hamilton'

start = time.perf_counter()
keywords = keywords_extractor(text, method='rake')

from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk import word_tokenize


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

expanded_keywords = thesaurus_based_expansion(text, keywords)
end = time.perf_counter()
print(f'Thesaurus-based Expansion Time: {end - start:.4f} seconds')
print('Expanded Keywords:', expanded_keywords)
