import sentencepiece as spm


def train_segmentation(text_file, LANG, vocab_size, model_type):
    """
    Objective:
        train a sentencepiece model to segment text into subwords or chars.
    Input:
        - text_file: a string with the path to the input text file.
        - LANG: the config for input language.
        - vocab_size: vocabulary size of the resulting segmentation. This parameter has no effect if the `model_type` is 'char'.
        - model_type: 'char' or 'bpe'.
    Output:
        a string with the path to the trained segmentation model.
    """
    if model_type == 'char':        
        model_prefix = f'{LANG.seg_model_folder}/spm_{LANG.name}_{model_type}'

        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=model_prefix,
            character_coverage=LANG.char_coverage,
            model_type=model_type,
        )

    elif model_type == 'bpe':        
        model_prefix = f'{LANG.seg_model_folder}/spm_{LANG.name}_{model_type}_vocabsize-{vocab_size}'

        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=LANG.char_coverage,
            model_type=model_type,
        )

    model_path = f'{model_prefix}.model'
    return model_path

def encode_text_file(text_file, model_path, output_file):
    """
    Objective:
        encode a text file (in which each sentence occupies a separate line) by a sentencepiece model.
    Input:
        - text_file: a string with the path to the input text file.
        - model_path: a string with the path to the sentencepiece model.
        - output_file: a string with path to the desired encoded text.
    Output:
        nothing.
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

def decode_text_file(text_file, model_path, output_file):
    """
    Objective:
        decode an encoded text file.
    Input:
        - text_file: a string with the path to the input encoded text file.
        - model_path: a string with the path to the sentencepiece model.
        - output_file: a string with path to the desired decoded text.
    Output:
        nothing.
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