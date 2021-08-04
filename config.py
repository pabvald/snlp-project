from nltk import tokenize


class Language:
    train_size = 0.8
    seg_model_folder = './seg_models'

class English(Language):
    name = 'en'
    raw_text_path = 'data/alice_in_wonderland.txt'
    vocab_size = 1000
    char_coverage = 1.0

    def sent_tokenize(text):
        return tokenize.sent_tokenize(text)

class Bengali(Language):
    name = 'bn'
    raw_text_path = 'data/bengali_corpus.txt'
    vocab_size = 300
    char_coverage = 0.995

    def sent_tokenize(text):
        #TODO: implement.
        pass