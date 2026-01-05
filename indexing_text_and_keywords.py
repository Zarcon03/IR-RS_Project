#NOTE: I BASICALLY JUST COPIED AND PASTED THE COLAB PARTS RELATED TO INDEXING
# TO CHANGE AND REDO IN ORDER FOR THE FIELDS TO BE CONSIDERED

import pyterrier as pt
import pandas as pd
import json
import shutil
import os
from functions import keywords_extractor, thesaurus_based_expansion
from tqdm import tqdm

tqdm.pandas()

#Get the dataframe to index
with open("document_collection.json", "r", encoding = "utf-8") as file:
    data = json.load (file)
df = pd.DataFrame.from_dict(data)


# Prepare dataframe for PyTerrier: needs columns 'docno' and 'text'
corpus_dataframe = df.rename(columns={"para_id": "docno", "context": "text"})[["docno", "text"]]
longest_len = corpus_dataframe["docno"].str.len().max()
print(corpus_dataframe["docno"])

#stats of normal text, wordcount
stats = corpus_dataframe["text"].str.split().str.len().agg( min_words="min", avg_words="mean", max_words="max", )
print(stats)


corpus_dataframe["keywords"] = corpus_dataframe["text"].progress_apply(
    lambda q: " ".join(
        thesaurus_based_expansion(q, keywords_extractor(q))
    )
)

#stats of keywords
stats = corpus_dataframe["keywords"].str.split().str.len().agg( min_words="min", avg_words="mean", max_words="max", )
print(stats)


# Create or reset an index folder
index_path = os.path.abspath("keywords_and_text_index")  # absolute path
if os.path.exists(index_path):
    shutil.rmtree(index_path)
os.makedirs(index_path, exist_ok=True)


# Build the indexer using the pt.IterDictIndexer 
# Store docno as metadata so we can recover it later if needed, do we remember why we did it?
# Key parameters now are: meta, text_attrs, meta_reverse, pretokenised, fields, threads
indexer = pt.IterDictIndexer(
    index_path,
    meta={"docno": longest_len},  #TO CHECK          # store docno as metadata (up to 200 characters)
    text_attrs=["text", "keywords"],           # which field(s) contain the text
    meta_reverse=["docno"],        # enable reverse lookup on docno
    pretokenised=False,
    threads=1, 
    fields=True
)



#perform the indexing and assign 
index_ref = indexer.index(corpus_dataframe.to_dict(orient="records"))
#index_ref is not the index itself, but a reference pointing to where the index was created

# Open the index to ensure it is valid
index = pt.IndexFactory.of(index_ref)

# Print a simple summary
print("Index location:", index_path)
print("Indexed documents:", index.getCollectionStatistics().getNumberOfDocuments())