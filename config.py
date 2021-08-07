from collections import namedtuple

# Paths 
figures_path = 'figures'
experiments_path = 'experiments'

# Neptune configuration
class NeptuneConfig:
  project = 'pabvald/snlp'
  api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNTIxNzcyMS04MTJiLTQ2ZWQtYWNmYy0wOWU1YjI0ZDU4NTIifQ=='

# Language configuration
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


# Segmentation Configurations
SegmentationConf = namedtuple('SegmentationConf', 'id vocab_size model_type')
seg_profiles = {
    'en': [
        SegmentationConf('s1', None, 'char'), # segmentation by characters.
        SegmentationConf('s2', 500, 'bpe'), # segmentation by subwords with small vocabulary.
        SegmentationConf('s3', 1500, 'bpe'), # segmentation by subwords with large vocabulary.
    ],
    'bn': [
        SegmentationConf('s1', None, 'char'), # segmentation by characters.
        SegmentationConf('s2', 800, 'bpe'), # segmentation by subwords with small vocabulary.
        SegmentationConf('s3', 1700, 'bpe'), # segmentation by subwords with large vocabulary.
    ]
}

# Training configurations
TrainingConf = namedtuple('TrainingConf', 'hidden rand_seed debug bptt n_class')
baseline_conf = TrainingConf(40, 1, 2, 4, 9999)
visualization_conf = TrainingConf(40, 1, 2, 3, 100)