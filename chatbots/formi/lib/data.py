import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
import re


MAX_UTTERANCE_LENGTH = 40
DATASET_PATH = "data/dataset"


def preprocess_utterance(utterance):
    utterance = utterance.lower().strip()
    utterance = re.sub(r"([?.!,]){2,}", r"\1", utterance)
    utterance = re.sub(r"([?.!,])", r" \1 ", utterance)
    utterance = re.sub(r"[^a-zA-Z?.!,]+", " ", utterance)
    utterance = utterance.strip()
    return utterance


def load_dataset():
    dataset = tf.data.Dataset.load(DATASET_PATH)

    dataset = dataset.cache()
    dataset = dataset.shuffle(20000)
    dataset = dataset.batch(64)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


class DataController:
    def __init__(self):
        self._tokenizer = None

    def get_tokenizer(self):
        if self._tokenizer == None:
            self._tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
                DATASET_PATH
            )
        return self._tokenizer

    def get_vocab_size(self):
        return self.get_tokenizer().vocab_size + 2

    def get_boundary_tokens(self):
        tokenizer = self.get_tokenizer()
        return tokenizer.vocab_size, tokenizer.vocab_size + 1

    def tokenize_and_filter(self, inputs, outputs):
        tokenizer = self.get_tokenizer()
        start_token, end_token = self.get_boundary_tokens()
        tokenized_inputs, tokenized_outputs = [], []

        for (i, o) in zip(inputs, outputs):
            # tokenize utterance
            i = [start_token] + tokenizer.encode(i) + [end_token]
            o = [start_token] + tokenizer.encode(o) + [end_token]
            # check tokenized utterance max length
            if len(i) <= MAX_UTTERANCE_LENGTH and len(o) <= MAX_UTTERANCE_LENGTH:
                tokenized_inputs.append(i)
                tokenized_outputs.append(o)

        # pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=MAX_UTTERANCE_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=MAX_UTTERANCE_LENGTH, padding='post')

        return tokenized_inputs, tokenized_outputs

    def save_dataset(self, inputs, outputs):
        print("> Exporting tokenizer vocabulary ...")

        self._tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            inputs + outputs,
            target_vocab_size=2**13,
        )
        self._tokenizer.save_to_file(DATASET_PATH)

        print("> Tokenizing data ...")

        inputs, outputs = self.tokenize_and_filter(inputs, outputs)

        print('Vocab size: {}'.format(self.get_vocab_size()))
        print('Number of samples: {}'.format(len(inputs)))

        # decoder inputs use the previous target as input
        # remove START_TOKEN from targets
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': inputs,
                'dec_inputs': outputs[:, :-1]
            },
            {
                'outputs': outputs[:, 1:]
            },
        ))

        print("> Saving dataset ...")

        if os.path.isdir(DATASET_PATH):
            shutil.rmtree(DATASET_PATH)
        dataset.save(DATASET_PATH)
