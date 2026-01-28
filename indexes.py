import torch
from collection import BenchmarkCollection
import pyterrier as pt
from tqdm import tqdm
from pathlib import Path
from functions import keywords_extractor, thesaurus_based_expansion
from constants import BASIC_INDEX_NAME, KEYWORDS_INDEX_NAME, TWO_FIELDS_INDEX_NAME, INDEXES_FOLDER, DENSE_INDEX_NAME, DEVICE
from pyterrier_dr import FlexIndex, RetroMAE
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BenchmarkIndex():
    def __init__(self, collection: BenchmarkCollection, indexes_folder=INDEXES_FOLDER):
        self.collection = collection
        self.indexes_folder = Path(indexes_folder).resolve()
        self._create_indexes_folder()
    
    def _create_indexes_folder(self):
        self.indexes_folder.mkdir(parents=True, exist_ok=True)

    def _ensure_index_folder(self, index_name: str) -> tuple[Path, bool]:
        path = self.indexes_folder / index_name
        try:
            path.mkdir(parents=True)
            logger.info("Created index directory: %s", path)
            return path, True
        except FileExistsError:
            logger.info("Index already exists: %s", path)
            return path, False
        
    def _build_indexer(self, index_path: Path, meta: dict, text_attrs: list[str], meta_reverse: list[str], pretokenised: bool, fields: bool, threads: int):
        return pt.IterDictIndexer(
            str(index_path),
            meta=meta,
            text_attrs=text_attrs,
            meta_reverse=meta_reverse,
            pretokenised=pretokenised,
            fields=fields,
            threads=threads
        )

    def create_basic_index(self):
        if not hasattr(self.collection, "corpus_dataframe"):
            raise RuntimeError("Documents not loaded. Call load_documents() before creating an index.")
        longest_len = self.collection.corpus_dataframe["docno"].str.len().max()
        longest_txt = self.collection.corpus_dataframe["text"].str.len().max()
        
        # Create index or raise error if it exists
        basic_index_path, created = self._ensure_index_folder(BASIC_INDEX_NAME)
        if not created:
            return
            
        # Build the indexer using the pt.IterDictIndexer
        indexer = self._build_indexer(
            index_path=basic_index_path,
            meta={"docno": longest_len, "text": longest_txt},
            text_attrs=["text"],
            meta_reverse=["docno"],
            pretokenised=False,
            fields=False,
            threads=1
        )

        index_ref = indexer.index(self.collection.corpus_dataframe.to_dict(orient="records"))

        # Open the index to ensure it is valid
        index = pt.IndexFactory.of(index_ref)

        # Print a simple summary
        print("Index location:", basic_index_path)
        print("Indexed documents:", index.getCollectionStatistics().getNumberOfDocuments())
    
    def create_keywords_expanded_index(self):
        if not hasattr(self.collection, "corpus_dataframe"):
            raise RuntimeError("Documents not loaded. Call load_documents() before creating an index.")
        longest_len = self.collection.corpus_dataframe["docno"].str.len().max()

        # Create index or raise error if it exists
        keywords_expanded_index_path, created = self._ensure_index_folder(KEYWORDS_INDEX_NAME)
        if not created:
            return

        corpus_dataframe = self.collection.corpus_dataframe.copy() # Make a copy to avoid modifying the original   
        # Expand documents with keywords and synonyms
        tqdm.pandas(desc="Expanding documents with keywords and synonyms")
        corpus_dataframe["text"] = corpus_dataframe["text"].progress_apply(
            lambda q: " ".join(
                thesaurus_based_expansion(q, keywords_extractor(q)),
            )
        )

        # Build the indexer using the pt.IterDictIndexer
        indexer = self._build_indexer(
            index_path=keywords_expanded_index_path,
            meta={"docno": longest_len},
            text_attrs=["text"],
            meta_reverse=["docno"],
            pretokenised=False,
            fields=False,
            threads=1
        )

        index_ref = indexer.index(corpus_dataframe.to_dict(orient="records"))
        
        # Open the index to ensure it is valid
        index = pt.IndexFactory.of(index_ref)

        # Print a simple summary
        print("Index location:", keywords_expanded_index_path)
        print("Indexed documents:", index.getCollectionStatistics().getNumberOfDocuments())
    
    def create_two_fields_index(self):
        if not hasattr(self.collection, "corpus_dataframe"):
            raise RuntimeError("Documents not loaded. Call load_documents() before creating an index.")
        longest_len = self.collection.corpus_dataframe["docno"].str.len().max()

        # Create index or raise error if it exists
        two_fields_index_path, created = self._ensure_index_folder(TWO_FIELDS_INDEX_NAME)
        if not created:
            return

        corpus_dataframe = self.collection.corpus_dataframe.copy() # Make a copy to avoid modifying the original
        # Create keywords field
        tqdm.pandas(desc="Creating keywords field")
        corpus_dataframe["keywords"] = corpus_dataframe["text"].progress_apply(
            lambda q: " ".join(
                thesaurus_based_expansion(q, keywords_extractor(q))
            )
        )

        # Build the indexer using the pt.IterDictIndexer
        indexer = self._build_indexer(
            index_path=two_fields_index_path,
            meta={"docno": longest_len},
            text_attrs=["text", "keywords"],
            meta_reverse=["docno"],
            pretokenised=False,
            fields=True,
            threads=1
        )
        # properties = {'index.document.class': 'FSADocumentIndexInMemFields'} # doesn't work

        index_ref = indexer.index(corpus_dataframe.to_dict(orient="records"))

        # Open the index to ensure it is valid
        index = pt.IndexFactory.of(index_ref)

        # Print a simple summary
        print("Index location:", two_fields_index_path)
        print("Indexed documents:", index.getCollectionStatistics().getNumberOfDocuments())

    def create_dense_index(self):
        if not hasattr(self.collection, "corpus_dataframe"):
            raise RuntimeError("Documents not loaded. Call load_documents() before creating an index.")
        
        # Create index or raise error if it exists
        dense_index_path = self.indexes_folder / DENSE_INDEX_NAME
        if dense_index_path.exists():
            logger.info(f"Index already exists: {DENSE_INDEX_NAME}")
            return

        # build an indexing pipeline that first applies RetroMAE to get dense vectors, then indexes them into the FlexIndex
        dense_index = FlexIndex(str(dense_index_path), verbose = 1)
        model = RetroMAE.msmarco_distill(device=DEVICE)
        offline_indexing_pipeline = model >> dense_index.indexer(mode="overwrite")
        
        corpus_dataframe = self.collection.corpus_dataframe.copy()
        corpus_dataframe = corpus_dataframe.to_dict(orient="records")
        # create the index
        offline_indexing_pipeline.index(corpus_dataframe)

        # Print a simple summary
        print("Index location:", dense_index_path)

    def _load_index(self, index_name: str):
        index_path = self.indexes_folder / index_name
        if not index_path.exists():
            raise RuntimeError(f"Index does not exist: {index_path}")
        try:
            return pt.IndexFactory.of(str(index_path))
        except:
            return FlexIndex(str(index_path))
        
    
    def load_basic_index(self):
        self.basic_index = self._load_index(BASIC_INDEX_NAME)
    
    def load_keywords_expanded_index(self):
        self.keywords_expanded_index = self._load_index(KEYWORDS_INDEX_NAME)

    def load_two_fields_index(self):
        self.two_fields_index = self._load_index(TWO_FIELDS_INDEX_NAME)

    def load_dense_index(self):
        self.dense_index = self._load_index(DENSE_INDEX_NAME)