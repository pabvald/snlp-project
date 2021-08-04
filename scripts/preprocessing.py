import re
from sklearn.model_selection import train_test_split

def raw_preprocess(LANG):
    """
    Objective: 
        preprocess the raw data so that it is ready for sentencepiece. In other words, the preprocessed data must have each sentence on one separated line.

    Assumptions:
        - Two-or-more end-of-line symbols (i.e. '\n\n') signify the separation of two paragraphs.
        - No single sentence resides in more than one paragraph.
        - We are allowed to use nltk for sentence-tokenization. The function from nltk (nltk.tokenize.sent_tokenize) is sufficient for the task.
        - There is no situation where double-space (i.e. '  ') makes sense, so we contract all sequences of more than one space into a space character (i.e. from the regex point of view, ' +' is substituted with ' ').

    Operations:
        - Get a list of paragraphs by splitting the text by at-least-2 consecutive end-of-line characters (i.e. '\n{2,}').
        - For each paragraph:
            + Replace all end-of-line characters with a space.
            + Contract all sequences of more than one space into a space character.
            + Tokenizer into sentences.
        - Collect all sentences from all paragraphs as a list and return.

    Input:
        LANG: the config for input language.

    Output:
        a list of extracted sentences.
    """
    raw_text_path = LANG.raw_text_path
    sentences = []

    # Read raw text file.
    with open(raw_text_path) as f:
        text = f.read()
    
    # Get a list of paragraphs by splitting the text by at-least-2 consecutive end-of-line characters (i.e. '\n{2,}').
    paragraphs = re.split(r'\n{2,}', text)

    # Extract sentences from each paragraph.
    for paragraph in paragraphs:
        # Replace all end-of-line characters with a space.
        paragraph = re.sub(r'\n', ' ', paragraph)
        # Contract all sequences of more than one space into a space character.
        paragraph = re.sub(r' +', ' ', paragraph).strip()
        # Tokenizer into sentences.
        paragraph_sentences = LANG.sent_tokenize(paragraph)
        # Collect all sentences from all paragraphs.
        sentences.extend(paragraph_sentences)

    return sentences

def split_train_test(sentences, LANG):
    """
    Objective:
        split the original data into train and test.

    Input:
        - sentences: a list of string, each string is a sentence.
        - LANG: the config for input language.

    Output:
        train and test sentences.
    """
    train, test = train_test_split(sentences, train_size=LANG.train_size, shuffle=True, random_state=42)
    return train, test

