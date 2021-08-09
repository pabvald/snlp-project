import nltk

def get_word_level_tokens(LANG, text_file):
    """
    Given a text file as input, tokenize the text on word-level and return the tokens.

    Parameters:
        LANG: the config for input language.
        text_file: a string contain the path to the text file.

    Returns:
        a list of tokens.
    """
    with open(text_file, 'r') as f:
        content = f.read()

    if LANG.name == 'en':
        tokens = nltk.word_tokenize(content)
    elif LANG.name == 'bn':
        pass #TODO
    
    return tokens

def get_word_level_vocabulary(LANG, text_file):
    """
    Given a text file as input, tokenize the text on word-level and return the vocabulary.

    Parameters:
        LANG: the config for input language.
        text_file: a string contain the path to the text file.

    Returns:
        a list of tokens.
    """
    tokens = get_word_level_tokens(LANG, text_file)
    vocab = set(tokens)
    return vocab

def get_OOV_rate(vocab, test_tokens):
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