from pyterrier.measures import P, R, nDCG, MAP
import torch

EVAL_METRICS = [P@1, P@5, P@10, R@5, R@10, nDCG@5, nDCG@10, MAP]

INDEXES_FOLDER = "indexes"
BASIC_INDEX_NAME = "basic_index"
KEYWORDS_INDEX_NAME = "keywords_expanded_index"
TWO_FIELDS_INDEX_NAME = "two_fields_index"
DENSE_INDEX_NAME = "dense_index.flex"
RESULTS_FOLDER = "results"

RANDOM_STATE = 42

# Determine device for model operations (e.g., 'xpu' for Intel GPUs, 'cuda' for NVIDIA GPUs)
DEVICE = 'xpu' if torch.xpu.is_available() else 'cpu'