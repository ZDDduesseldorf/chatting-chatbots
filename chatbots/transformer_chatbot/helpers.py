
from tqdm import tqdm
import tensorflow_datasets as tfds
import tensorflow as tf
import re
import pandas as pd
from dotenv import load_dotenv
import os
import pickle
load_dotenv()

BUFFER_SIZE = int(os.environ.get('BUFFER_SIZE'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
EPOCHS = int(os.environ.get('EPOCHS'))
directory = './data/'
path = f"{directory}{EPOCHS}EPOCHS_{MAX_SAMPLES}SAMPLES_{MAX_LENGTH}LENGTH/"


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

    if MAX_SAMPLES == 0:
        input_list = input_df.values.tolist()
        output_list = output_df.values.tolist()
    else:
        input_list = input_df.values.tolist()[:MAX_SAMPLES]
        output_list = output_df.values.tolist()[:MAX_SAMPLES]
    
    preprocessed_inputs, preprocessed_outputs = [], []
    progress = tqdm(range(len(input_list) - 1))
    for index in progress:
        progress.set_description('Reading from csv')
        input = input_list[index]
        output = output_list[index]
        
        if type(input) == str and type(output) == str:
            # first preprocess then check for length?!
            if len(input.split()) <= MAX_LENGTH and len(output.split()) <= MAX_LENGTH:
                preprocessed_inputs.append(preprocess_sentence(input))
                preprocessed_outputs.append(preprocess_sentence(output))
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
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


def get_tokenizer():
    return tfds.deprecated.text.SubwordTextEncoder.load_from_file(filename_prefix=f"{path}tokenizer")

def create_and_save_dataset(x, y, name):
    #set validation dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': x,
            'dec_inputs': y[:, :-1]
        },
        {
            'outputs': y[:, 1:]
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    tf.data.experimental.save(
        dataset, path + name
    )
    with open(path + name + '/element_spec', 'wb') as out_:
        pickle.dump(dataset.element_spec, out_)

def load_dataset(name):
    with open(path + name + '/element_spec', 'rb') as in_:
        es = pickle.load(in_)
    return tf.data.experimental.load(path + name, es)