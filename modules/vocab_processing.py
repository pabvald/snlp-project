import nltk
from bltk.langtools import Tokenizer as BengaliTokenizer
from config import Language
from typing import List, Set

from preprocessing import text_preprocess

def get_word_level_tokens(LANG: Language, text_file: str):
    """
    Given a text file as input, tokenize the text on word-level and return the tokens.

    Parameters:
        LANG: the config for input language.
        text_file: path to the text file.

    Returns:
        a list of tokens.
    """
    with open(text_file, 'r') as f:
        text = f.read()

    preprocessed_text = text_preprocess(LANG, text)

    if LANG.name == 'en':
        tokens = nltk.word_tokenize(preprocessed_text)
    elif LANG.name == 'bn':
        tokenizer = BengaliTokenizer()
        tokens = tokenizer.word_tokenizer(preprocessed_text)
    
    return tokens

def get_word_level_vocabulary(LANG: Language, text_file: str):
    """
    Given a text file as input, tokenize the text on word-level and return the vocabulary.

    Parameters:
        LANG: the config for input language.
        text_file: path to the text file.

    Returns:
        a list of tokens.
    """
    tokens = get_word_level_tokens(LANG, text_file)
    vocab = set(tokens)
    return vocab

def get_oov_rate(vocab: Set[str], test_tokens: List[str]):
    """
    Compute the OOV rate.

    Parameters:
        vocab: a set of token types.
        test_tokens: a list of test tokens.

    Returns:
        a floating-point number as the OOV rate.
    """
    oov_tokens = [token for token in test_tokens if token not in vocab]
    oov_rate = len(oov_tokens) / len(test_tokens)
    return oov_rate