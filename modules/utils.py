import numpy as np
from typing import List


def print_length_statistics(sentences: List[str]):
    """ 
    Compute some statistics of the length of the sentences.

    Parametters:
        sentences: list of sentences
    """
    lengths = [len(s) for s in sentences]
    print(f"Num. of sentences = {len(lengths)}")
    print(f"Avg. length = {np.mean(lengths)}")
    print(f"Std. dev. of the length = {np.std(lengths)}")
    print(f"Maximum length = {max(lengths)}")
    print(f"Minimum length = {min(lengths)}")