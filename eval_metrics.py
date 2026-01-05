from pyterrier.measures import P, R, nDCG, MAP
EVAL_METRICS = [P@1, P@5, P@10, R@5, R@10, nDCG@5, nDCG@10, MAP]