import helpers
from helpers import Params
from tqdm import tqdm
import tensorflow_datasets as tfds
import tensorflow as tf
import re
import pandas as pd
import os
import pickle
from halo import Halo


def preprocess_sentence(sentence: str):
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


def load_conversations(params: Params, max_samples: int = None):
    if (max_samples == None):
        max_samples = params.max_samples

    path = f"{params.data_root}/{params.dataset}.csv"
    print(f"Loading conversations from {path}")
    df = pd.read_csv(path, sep=';')
    input_df = df['Input']
    output_df = df['Output']

    if max_samples == 0:
        input_list = input_df.values.tolist()
        output_list = output_df.values.tolist()
    else:
        input_list = input_df.values.tolist()[:max_samples]
        output_list = output_df.values.tolist()[:max_samples]

    preprocessed_inputs, preprocessed_outputs = [], []
    progress = tqdm(range(len(input_list)))
    for index in progress:
        progress.set_description('Reading from csv')
        input = input_list[index]
        output = output_list[index]
        if type(input) == str and type(output) == str:
            preprocessed_inputs.append(preprocess_sentence(input))
            preprocessed_outputs.append(preprocess_sentence(output))

    return preprocessed_inputs, preprocessed_outputs


def tokenize_and_filter(params: Params, inputs, outputs, tokenizer):
    tokenized_inputs, tokenized_outputs = [], []
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        if len(sentence1) <= params.max_length and len(sentence2) <= params.max_length:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

        # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=params.max_length, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=params.max_length, padding='post')

    return tokenized_inputs, tokenized_outputs


def get_start_and_end_tokens(params: Params):
    tokenizer = get_tokenizer(params)
    return [tokenizer.vocab_size], [tokenizer.vocab_size + 1]


def get_vocab_size(params: Params):
    return get_tokenizer(params).vocab_size + 2


def get_tokenizer(params: Params):
    if os.path.exists(f"{params.tokenizer_path}.subwords"):
        spinner = Halo(
            text=f"Loading Tokenizer from {params.tokenizer_path}", spinner='dots')
        spinner.start()
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
            params.tokenizer_path)
    else:
        helpers.ensure_dir(params.tokenizer_dir)
        # Build tokenizer using tfds for both questions and answers
        questions, answers = load_conversations(params, 0)
        spinner = Halo(
            text='Creating tokenizer and vocabulary ...', spinner='dots')
        spinner.start()
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=params.target_vocab_size)
        tokenizer.save_to_file(params.tokenizer_path)

    spinner.stop()

    return tokenizer


def create_and_save_dataset(params: Params, x, y, name):
    # set validation dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'inputs': x, 'dec_inputs': y[:, :-1]}, {'outputs': y[:, 1:]}))

    dataset = dataset.cache()
    dataset = dataset.shuffle(params.buffer_size)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    dir = f"{params.dataset_dir}/{name}"
    helpers.ensure_dir(dir)
    tf.data.experimental.save(dataset, dir)
    with open(f"{dir}/element_spec", 'wb') as out_:
        pickle.dump(dataset.element_spec, out_)

    return dataset


def load_dataset(params: Params, name):
    dir = f"{params.dataset_dir}/{name}"
    with open(f"{dir}/element_spec", 'rb') as in_:
        es = pickle.load(in_)
    return tf.data.experimental.load(dir, es)


def init_datasets(params: Params):
    tf.random.set_seed(1234)

    questions, answers = load_conversations(params)

    tokenizer = get_tokenizer(params)

    spinner = Halo(
        text='Tokenize and filter dataset sentences ...', spinner='dots')
    spinner.start()

    # split dataset 70:30
    val_split = int(len(questions)*(9/10))
    train_questions, train_answers = tokenize_and_filter(
        params, questions[:val_split], answers[:val_split], tokenizer)
    val_questions, val_answers = tokenize_and_filter(
        params, questions[val_split:], answers[val_split:], tokenizer)

    # split into train and test data
    spinner.stop()

    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    spinner = Halo(text='Create tensors from dataset ...', spinner='dots')
    spinner.start()
    train_dataset = create_and_save_dataset(
        params, train_questions, train_answers, "train")
    val_dataset = create_and_save_dataset(
        params, val_questions, val_answers, "val")
    spinner.stop()

    print('Size training samples:', len(train_questions))
    print('Size val samples:', len(val_questions))

    return train_dataset, val_dataset


def get_datasets(params: Params):
    if os.path.exists(params.dataset_dir):
        return load_dataset(params, "train"), load_dataset(params, "val")
    else:
        return init_datasets(params)
