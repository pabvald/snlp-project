import sentencepiece as spm
from typing import List
from config import Language, SEG_MODELS_PATH


def train_segmentation(text_file: str, LANG: Language, vocab_size: int, model_type: str):
    """
    Train a sentencepiece model to segment text into subwords or chars.

    Parameters:
        text_file: path to the input text file.
        LANG: the config for input language.
        vocab_size: vocabulary size of the resulting segmentation. This parameter has no effect if the `model_type` is 'char'.
        model_type: 'char' or 'bpe'.

    Returns:
        path to the trained segmentation model.
    """
    assert model_type in {'char', 'bpe'}, 'invalid model type'

    if model_type == 'char':        
        model_prefix = f'{SEG_MODELS_PATH}/{LANG.name}/spm_{model_type}'

        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=model_prefix,
            character_coverage=LANG.char_coverage,
            model_type=model_type,
        )

    elif model_type == 'bpe':        
        model_prefix = f'{SEG_MODELS_PATH}/{LANG.name}/spm_{model_type}_vocabsize-{vocab_size}'

        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=LANG.char_coverage,
            model_type=model_type,
        )

    model_path = f'{model_prefix}.model'
    return model_path

def encode_text_file(text_file: str, model_path: str, output_file: str):
    """
    Encode a text file (in which each sentence occupies a separate line) by a sentencepiece model.
    
    Parameters:
        text_file: path to the input text file.
        model_path: the path to the sentencepiece model.
        output_file: path to the desired encoded text.
    """

    # load model.
    sp = spm.SentencePieceProcessor(model_file=model_path)

    # load input text.
    with open(text_file, 'r') as in_file:
        lines = in_file.readlines()

    # encode.
    encoded_sentences = [' '.join(sp.encode(line, out_type='str')) \
        for line in lines]

    # write output to file.
    with open(output_file, 'w') as out_file:
        out_file.write('\n'.join(encoded_sentences))

def decode_text_file(text_file: str, model_path: str, output_file: str):
    """
    Decode an encoded text file.

    Parameters:
        text_file: the path to the input encoded text file.
        model_path: the path to the sentencepiece model.
        output_file: path to the desired decoded text.
    """

    # load model.
    sp = spm.SentencePieceProcessor(model_file=model_path)

    # load input text.
    with open(text_file, 'r') as in_file:
        lines = in_file.readlines()

    # decode.
    decoded_sentences = [sp.decode(line.rstrip().split(' ')) for line in lines]

    # write output to file.
    with open(output_file, 'w') as out_file:
        out_file.write('\n'.join(decoded_sentences))