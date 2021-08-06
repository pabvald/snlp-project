import re
from nltk import tokenize
from bltk.langtools import Tokenizer as BengaliTokenizer
from sklearn.model_selection import train_test_split

def raw_preprocess(LANG):
    """ 
    Preprocesses the raw data so that it is ready for sentencepiece. In other words, the 
    preprocessed data must have each sentence on one separated line.
    
    Parameters:
        LANG: the config for input language.

    Return:
        a list of extracted sentences.
    """
    raw_text_path = LANG.raw_text_path

    # Read raw text file.
    with open(raw_text_path) as f:
        text = f.read()
    # Get a list of paragraphs by splitting the text by at-least-2 
    # consecutive end-of-line characters (i.e. '\n{2,}').
    paragraphs = re.split(r'\n{2,}', text)

    # Apply preprocessing 
    if LANG.name == 'en':
        sentences = _raw_preprocess_english(paragraphs)
    elif LANG.name == 'bn':
        sentences = _raw_preprocess_bengali(paragraphs)
    else: 
        raise ValueError('preprocessing for this language is  not implemented')

    return sentences

def _raw_preprocess_english(paragraphs):
    """
    Preprocesses a list of paragraphs of the English corpus.

    Parameters:
       paragraphs: list of paragraphs

    Returns:
        a list of extracted sentences.
    """
    sentences = []
    
    # Extract sentences from each paragraph.
    for paragraph in paragraphs:
        # Replace all end-of-line characters with a space.
        paragraph = re.sub(r'\n', ' ', paragraph)
        # Contract all sequences of more than one space into a space character.
        paragraph = re.sub(r' +', ' ', paragraph).strip()
        # Tokenizer into sentences.
        paragraph_sentences = tokenize.sent_tokenize(paragraph, language='english')
        # Collect all sentences from all paragraphs.
        sentences.extend(paragraph_sentences)

    return sentences

def _raw_preprocess_bengali(paragraphs):
    """ 
   Preprocesses a list of paragraphs of the Bengali corpus.
    
    Parameters:
        paragraphs: list of paragraphs

    Returns:
        a list of extracted sentences 
    """
    sentences = []
    
    for paragraph in paragraphs:
        # Remove HTML tags
        paragraph = re.sub('<.*?>', '', paragraph)
        # Substitute two or more exclamations/interrogations/full stops by a single one
        paragraph = re.sub(r'\?{2,}', '?', paragraph)
        paragraph = re.sub(r'\!{2,}', '!', paragraph)
        paragraph = re.sub(r'।{2,}', '|', paragraph)    
        # Remove English text 
        paragraph = re.sub(r'[a-zA-Z]', '', paragraph)    
        # Tokenize into sentences
        tokenizer = BengaliTokenizer()
        paragraph_sentences = tokenizer.sentence_tokenizer(paragraph)
        # Collect all sentences from all paragraphs
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

