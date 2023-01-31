import os
import pickle
import sentence_processing
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
    """Get START and END tokens from persisted tokenizer (both are not included in tokenizer vocabulary)."""
    tokenizer = get_tokenizer()
    return [tokenizer.vocab_size], [tokenizer.vocab_size + 1]


def get_vocab_size():
    """Get vocab size from persisted tokenizer including START and END token."""
    tokenizer = get_tokenizer()
    return tokenizer.vocab_size + 2


def load_conversations(datapath, filename):
    """Load question/answer pairs from csv containing 2M samples."""
    df = pd.read_csv(datapath + filename, sep=";")
    input_df = df["Input"]
    output_df = df["Output"]

    # defaults to 0 when whole dataset is respected
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
        # samples not of type str are filtered out
        if type(input) == str and type(output) == str:
            input = sentence_processing.preprocess_sentence(input)
            output = sentence_processing.preprocess_sentence(output)
            max_sentence_length = MAX_LENGTH - 2
            input, appendInput = sentence_processing.trim_sentence(input, max_sentence_length)
            output, appendOutput = sentence_processing.trim_sentence(output, max_sentence_length)                   
            if appendOutput and appendInput:
                preprocessed_inputs.append(input)
                preprocessed_outputs.append(output)
    return preprocessed_inputs, preprocessed_outputs

def tokenize_and_filter(inputs, outputs):
    """Add START and END token to dataset question/answer samples. Tokenize words and provide padding."""
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
    """Load tokenizer from file."""
    return tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        filename_prefix="chatbot_model/tokenizer"
    )


def create_tokenizer(questions, answers):
    """Create tokenizer and store file."""
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2**14
    )
    tokenizer.save_to_file(filename_prefix=f"{path}tokenizer")
    return tokenizer


def create_and_save_dataset(x, y, name):
    """Create and save Tensorflow Dataset."""
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
    """Load Tensorflow dataset."""
    with open(path + name + "/element_spec", "rb") as in_:
        es = pickle.load(in_)
    return tf.data.experimental.load(path + name, es)
