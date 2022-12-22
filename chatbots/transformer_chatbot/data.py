import helpers
from helpers import Params
from lib.tokenizer import TransformerTokenizer
from tqdm import tqdm
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

    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    # adding a start and an end token to the sentence
    return sentence


class DataLoader:
    def __init__(self, params: Params):
        self.params = params
        self._datasets: dict[str, tf.data.Dataset] = {}
        self._tokenizer: TransformerTokenizer = None
        self._conversations: tuple[list, list] = None
        self._samples: int = None

    def _load_tokenizer(self):
        spinner = Halo(
            text=f"Loading Tokenizer from {self.params.tokenizer_path}", spinner='dots')
        spinner.start()

        self._tokenizer = TransformerTokenizer.load_from_file(
            self.params.tokenizer_path)

        spinner.stop()

    def _build_tokenizer(self):
        inputs, outputs = self.get_conversations(0)

        spinner = Halo(
            text='Creating tokenizer and vocabulary ...',
            spinner='dots')
        spinner.start()

        self._tokenizer = TransformerTokenizer.build_from_corpus(
            inputs + outputs,
            self.params.target_vocab_size)

        helpers.ensure_dir(self.params.tokenizer_dir)
        self._tokenizer.save_to_file(self.params.tokenizer_path)

        spinner.stop()

    def get_tokenizer(self):
        if self._tokenizer is None:
            if os.path.exists(f"{self.params.tokenizer_path}.subwords"):
                self._load_tokenizer()
            else:
                self._build_tokenizer()
        return self._tokenizer

    def _tokenize_and_filter(self, inputs, outputs):
        tokenizer = self.get_tokenizer()
        tokenized_inputs, tokenized_outputs = [], []
        START_TOKEN, END_TOKEN = [tokenizer.start_token], [tokenizer.end_token]

        for (sentence1, sentence2) in zip(inputs, outputs):
            # tokenize sentence
            sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
            if len(sentence1) <= self.params.max_length and len(sentence2) <= self.params.max_length:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)

            # pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=self.params.max_length, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=self.params.max_length, padding='post')

        return tokenized_inputs, tokenized_outputs

    def get_conversations(self, max_samples: int = None):
        if (max_samples is None):
            max_samples = self.params.max_samples

        if self._samples is None or (self._samples != 0 and self._samples < max_samples):
            path = f"{self.params.data_root}/{self.params.dataset}.csv"
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

            self._samples = max_samples
            self._conversations = (preprocessed_inputs, preprocessed_outputs)

        return self._conversations

    def _create_and_save_dataset(self, x: list, y: list, name: str):
        dataset = tf.data.Dataset.from_tensor_slices(
            ({'inputs': x, 'dec_inputs': y[:, :-1]}, {'outputs': y[:, 1:]}))

        dataset = dataset.cache()
        dataset = dataset.shuffle(self.params.buffer_size)
        dataset = dataset.batch(self.params.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        dir = f"{self.params.dataset_dir}/{name}"
        helpers.ensure_dir(dir)
        tf.data.experimental.save(dataset, dir)
        with open(f"{dir}/element_spec", 'wb') as out_:
            pickle.dump(dataset.element_spec, out_)

        return dataset

    def _init_datasets(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        tf.random.set_seed(1234)

        self.get_tokenizer()
        questions, answers = self.get_conversations()

        spinner = Halo(
            text='Tokenize and filter dataset sentences ...', spinner='dots')
        spinner.start()

        val_split = int(len(questions)*(9/10))
        train_questions, train_answers = self._tokenize_and_filter(
            questions[:val_split], answers[:val_split])
        val_questions, val_answers = self._tokenize_and_filter(
            questions[val_split:], answers[val_split:])

        spinner.stop()

        spinner = Halo(text='Create tensors from dataset ...', spinner='dots')
        spinner.start()
        train_dataset = self._create_and_save_dataset(
            train_questions, train_answers, "train")
        val_dataset = self._create_and_save_dataset(
            val_questions, val_answers, "val")
        spinner.stop()

        print('Size training samples:', len(train_questions))
        print('Size val samples:', len(val_questions))

        return train_dataset, val_dataset

    def get_dataset(self, name: str) -> tf.data.Dataset:
        if name not in self._datasets:
            dir = f"{self.params.dataset_dir}/{name}"
            with open(f"{dir}/element_spec", 'rb') as f:
                es = pickle.load(f)
            self._datasets[name] = tf.data.experimental.load(dir, es)
        return self._datasets[name]

    def get_datasets(self):
        if os.path.exists(self.params.dataset_dir):
            return self.get_dataset("train"), self.get_dataset("val")
        else:
            return self._init_datasets()
