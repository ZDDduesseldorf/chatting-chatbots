
from tqdm import tqdm
import tensorflow_datasets as tfds
import tensorflow as tf
import re
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()

MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))


def get_start_and_end_tokens():
    tokenizer = get_tokenizer()
    return [tokenizer.vocab_size], [tokenizer.vocab_size + 1]


def get_vocab_size():
    tokenizer = get_tokenizer()
    return tokenizer.vocab_size + 2


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    # adding a start and an end token to the sentence
    return sentence


def load_conversations(datapath, filename):
    df = pd.read_csv(datapath+filename, sep=';')
    input_df = df['Input']
    output_df = df['Output']
    input_list = input_df.values.tolist()
    output_list = output_df.values.tolist()
    preprocessed_inputs, preprocessed_outputs = [], []
    progress = tqdm(range(len(input_list) - 1))
    for index in progress:
        progress.set_description('Reading from csv')
        if type(input_list[index]) == str and type(output_list[index]) == str:
            preprocessed_inputs.append(preprocess_sentence(input_list[index]))
            preprocessed_outputs.append(
                preprocess_sentence(output_list[index]))
    return preprocessed_inputs, preprocessed_outputs


def tokenize_and_filter(inputs, outputs):
    tokenizer = get_tokenizer()
    tokenized_inputs, tokenized_outputs = [], []
    START_TOKEN, END_TOKEN = get_start_and_end_tokens()

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

        # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


def get_tokenizer():
    return tfds.deprecated.text.SubwordTextEncoder.load_from_file(filename_prefix='tokenizer')
