from nltk.corpus import wordnet as wn

print(wn.synonyms("bank"))

def expand_keywords_wordnet(keywords, n):
    """
    Expand a list of keywords using WordNet synonyms.

    Parameters
    keywords : list[str]
    n : int

    Returns
    list[str]
    """
    expanded = []

    for keyword in keywords:
        synonyms = set()

        for synset in wn.synonyms(keyword):
            print(synset)
            for lemma in synset:
                name = lemma.replace("_", " ")
                if name.lower() != keyword.lower():
                    print(name, keyword)
                    synonyms.add(name)

            if len(synonyms) >= n:
                break

        expanded.extend(list(synonyms)[:n])

    return expanded