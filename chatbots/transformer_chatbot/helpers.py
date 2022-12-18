from tqdm import tqdm
import tensorflow_datasets as tfds
import tensorflow as tf
import re
import pandas as pd
from dotenv import load_dotenv
import os
import pickle
from halo import Halo
load_dotenv()


DATASET_NAME = os.environ.get('DATASET')
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
TARGET_VOCAB_SIZE = 2**int(os.environ.get('VOCAB_SIZE_EXPONENT'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
BUFFER_SIZE = int(os.environ.get('BUFFER_SIZE'))
NUM_LAYERS = int(os.environ.get('NUM_LAYERS'))
NUM_HEADS = int(os.environ.get('NUM_HEADS'))
EPOCHS = int(os.environ.get('EPOCHS'))
D_MODEL = int(os.environ.get('D_MODEL'))
UNITS = int(os.environ.get('UNITS'))
DROPOUT = float(os.environ.get('DROPOUT'))

with open('.env', 'r') as f:
    print(f.read())


def ensure_dir(dir):
    if os.path.exists(dir):
        print(f"Directory {dir} already exitsts")
        return True
    else:
        print(f"Directory {dir} created")
        os.makedirs(dir)
        return False


TOKENIZER_KEY = f"{DATASET_NAME}/{TARGET_VOCAB_SIZE}Voc"
DATASET_KEY = f"{TOKENIZER_KEY}/{MAX_SAMPLES}Smp_{MAX_LENGTH}Len_{BATCH_SIZE}Bat_{BUFFER_SIZE}Buf"
MODEL_KEY = f"{DATASET_KEY}/{NUM_LAYERS}Lay_{NUM_HEADS}Hed_{EPOCHS}Epo"
LOGS_KEY = MODEL_KEY.replace('/', '__')

DATA_DIR = "data"
TOKENIZER_DIR = f"{DATA_DIR}/{TOKENIZER_KEY}"
TOKENIZER_PATH = f"{TOKENIZER_DIR}/tokenizer"
DATASET_DIR = f"{DATA_DIR}/{DATASET_KEY}"
MODEL_DIR = f"{DATA_DIR}/{MODEL_KEY}"
WEIGHTS_PATH = f"{MODEL_DIR}/weights"

LOGS_DIR = f"logs/{LOGS_KEY}"

ensure_dir(DATA_DIR)


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


def load_conversations(max_samples):
    print(
        f"Loading conversations from {DATA_DIR}/{DATASET_NAME}.csv")
    df = pd.read_csv(f"{DATA_DIR}/{DATASET_NAME}.csv", sep=';')
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
        preprocessed_inputs.append(preprocess_sentence(input_list[index]))
        preprocessed_outputs.append(preprocess_sentence(output_list[index]))

    return preprocessed_inputs, preprocessed_outputs


def tokenize_and_filter(inputs, outputs, tokenizer):
    tokenized_inputs, tokenized_outputs = [], []
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

        # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


def get_start_and_end_tokens():
    tokenizer = get_tokenizer()
    return [tokenizer.vocab_size], [tokenizer.vocab_size + 1]


def get_vocab_size():
    tokenizer = get_tokenizer()
    return tokenizer.vocab_size + 2


def get_tokenizer():
    if os.path.exists(f"{TOKENIZER_PATH}.subwords"):
        spinner = Halo(
            text=f"Loading Tokenizer from {TOKENIZER_PATH}", spinner='dots')
        spinner.start()
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
            TOKENIZER_PATH)
    else:
        ensure_dir(TOKENIZER_DIR)
        # Build tokenizer using tfds for both questions and answers
        questions, answers = load_conversations(0)
        spinner = Halo(
            text='Creating tokenizer and vocabulary ...', spinner='dots')
        spinner.start()
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=TARGET_VOCAB_SIZE)
        tokenizer.save_to_file(TOKENIZER_PATH)

    spinner.stop()

    return tokenizer


def create_and_save_dataset(x, y, name):
    # set validation dataset
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

    dataset_save = f"{DATASET_DIR}/{name}"
    ensure_dir(dataset_save)
    tf.data.experimental.save(dataset, dataset_save)
    with open(f"{dataset_save}/element_spec", 'wb') as out_:
        pickle.dump(dataset.element_spec, out_)

    return dataset


def load_dataset(name):
    dataset_save = f"{DATASET_DIR}/{name}"
    with open(f"{dataset_save}/element_spec", 'rb') as in_:
        es = pickle.load(in_)
    return tf.data.experimental.load(dataset_save, es)


def init_datasets():
    tf.random.set_seed(1234)

    questions, answers = load_conversations(MAX_SAMPLES)

    tokenizer = get_tokenizer()

    spinner = Halo(
        text='Tokenize and filter dataset sentences ...', spinner='dots')
    spinner.start()

    # split dataset 70:30
    val_split = int(len(questions)*(2/3))
    train_questions, train_answers = tokenize_and_filter(
        questions[:val_split], answers[:val_split], tokenizer)
    val_questions, val_answers = tokenize_and_filter(
        questions[val_split:], answers[val_split:], tokenizer)

    # split into train and test data
    spinner.stop()

    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    spinner = Halo(text='Create tensors from dataset ...', spinner='dots')
    spinner.start()
    train_dataset = create_and_save_dataset(
        train_questions, train_answers, "train")
    val_dataset = create_and_save_dataset(
        val_questions, val_answers, "val")
    spinner.stop()

    print('Size training samples:', len(train_questions))
    print('Size val samples:', len(val_questions))

    return train_dataset, val_dataset


def get_datasets():
    if ensure_dir(DATASET_DIR):
        return load_dataset("train"), load_dataset("val")
    else:
        return init_datasets()
