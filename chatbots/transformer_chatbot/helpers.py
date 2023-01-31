import os
import pickle
import re

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BUFFER_SIZE = int(os.environ.get('BUFFER_SIZE'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
MIN_LENGTH = int(os.environ.get('MIN_LENGTH'))
EPOCHS = int(os.environ.get('EPOCHS'))
directory = './data/merged/'
path = f"{directory}{MAX_LENGTH}LENGTH/"


def get_start_and_end_tokens():
    tokenizer = get_tokenizer()
    return [tokenizer.vocab_size], [tokenizer.vocab_size + 1]


def get_vocab_size():
    tokenizer = get_tokenizer()
    return tokenizer.vocab_size + 2


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # set dot at the end of sentence if there is no ?.!
    if re.search('[.!?]$',sentence) is None:
        sentence = sentence + '.'
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", sentence)
    sentence = sentence.strip()
    
    return sentence


def load_conversations(datapath, filename):
    df = pd.read_csv(datapath + filename, sep=";")
    input_df = df["Input"]
    output_df = df["Output"]

    if MAX_SAMPLES == 0:
        input_list = input_df.values.tolist()
        output_list = output_df.values.tolist()
    else:
        input_list = input_df.values.tolist()[:MAX_SAMPLES]
        output_list = output_df.values.tolist()[:MAX_SAMPLES]

    preprocessed_inputs, preprocessed_outputs = [], []
    progress = tqdm(range(len(input_list) - 1))
    for index in progress:
        progress.set_description("Reading from csv")
        input = input_list[index]
        output = output_list[index]

        if type(input) == str and type(output) == str:
            input = preprocess_sentence(input)
            output = preprocess_sentence(output)
            max_sentence_length = MAX_LENGTH - 2
            output_words = output.split()
            input_words = input.split()
            appendOutput = True
            appendInput = True
            if len(output_words) > max_sentence_length:
                output_words = output_words[:max_sentence_length-1]
                appendOutput = False
                if "?" in output_words:
                    index = output_words.index("?")
                    if index > 0:
                        output_words = output_words[:index+1]
                        output = " ".join(output_words)
                        appendOutput = True
                else:
                    if "!" in output_words:
                        index = output_words.index("!")
                        if index > 0:
                            output_words = output_words[:index+1]
                            output = " ".join(output_words)
                            appendOutput = True
                    else:
                        if "." in output_words:
                            index = output_words.index(".")
                            if index > 0 and output_words[index-1] != 'www':
                                output_words = output_words[:index+1]
                                output = " ".join(output_words)
                                appendOutput = True
            if len(input_words) > max_sentence_length:
                input_words = input_words[:max_sentence_length-1]
                appendInput = False
                if "?" in input_words:
                    index = input_words.index("?")
                    if index > 0:
                        input_words = input_words[:index+1]
                        input = " ".join(input_words)
                        appendInput = True
                else:
                    if "!" in input_words:
                        index = input_words.index("!")
                        if index > 0:
                            input_words = input_words[:index+1]
                            input = " ".join(input_words)
                            appendInput = True
                    else:
                        if "." in input_words:
                            index = input_words.index(".")
                            if index > 0 and input_words[index-1] != 'www':
                                input_words = input_words[:index+1]
                                input = " ".join(input_words)
                                appendInput = True                                
            if appendOutput and appendInput:
                preprocessed_inputs.append(input)
                preprocessed_outputs.append(output)
    return preprocessed_inputs, preprocessed_outputs


def tokenize_and_filter(inputs, outputs):
    tokenizer = get_tokenizer()
    tokenized_inputs, tokenized_outputs = [], []
    START_TOKEN, END_TOKEN = get_start_and_end_tokens()

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)

        # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding="post"
    )
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding="post"
    )

    return tokenized_inputs, tokenized_outputs


def get_tokenizer():
    return tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        filename_prefix="chatbot_model/tokenizer"
    )


def create_tokenizer(questions, answers):
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2**14
    )
    tokenizer.save_to_file(filename_prefix=f"{path}tokenizer")
    return tokenizer


def create_and_save_dataset(x, y, name):
    # set validation dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"inputs": x, "dec_inputs": y[:, :-1]},
            {"outputs": y[:, 1:]},
        )
    )

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    tf.data.experimental.save(dataset, path + name)
    with open(path + name + "/element_spec", "wb") as out_:
        pickle.dump(dataset.element_spec, out_)


def load_dataset(name):
    with open(path + name + "/element_spec", "rb") as in_:
        es = pickle.load(in_)
    return tf.data.experimental.load(path + name, es)
