

class NeptuneConfig:
  project = 'pabvald/snlp'
  api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNTIxNzcyMS04MTJiLTQ2ZWQtYWNmYy0wOWU1YjI0ZDU4NTIifQ=='

class Language:
    train_size = 0.8
    seg_model_folder = './seg_models'

class English(Language):
    name = 'en'
    raw_text_path = 'data/alice_in_wonderland.txt'
    vocab_size = 1000
    char_coverage = 1.0

class Bengali(Language):
    name = 'bn'
    raw_text_path = 'data/bengali_corpus.txt'
    vocab_size = 300
    char_coverage = 0.995