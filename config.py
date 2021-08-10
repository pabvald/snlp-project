from collections import namedtuple

# Paths 
SEG_MODELS_PATH = 'seg_models'
FIGURES_PATH = 'figures'
EXPERIMENTS_PATH = 'experiments'

# Language configuration
class Language:
    train_size = 0.8

class English(Language):
    name = 'en'
    raw_text_path = 'data/en/alice_in_wonderland.txt'
    vocab_size = 1000
    char_coverage = 1.0

class Bengali(Language):
    name = 'bn'
    raw_text_path = 'data/bn/bengali_corpus.txt'
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
optimal_conf = {
    'en': [
        TrainingConf(200, 1, 2, 3, 100),
        TrainingConf(200, 1, 2, 3, 1501),
        TrainingConf(40, 1, 2, 0, 1501),
    ], 
    'bn': [
        TrainingConf(200, 1, 2, 4, 1700),
        TrainingConf(200, 1, 2, 4, 1700),
        TrainingConf(200, 1, 2, 4, 1700),
    ], 
}

visualization_conf = TrainingConf(40, 1, 2, 3, 100)

