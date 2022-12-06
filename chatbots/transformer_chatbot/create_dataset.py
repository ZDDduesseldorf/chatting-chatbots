from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as ks
from nltk.tokenize import word_tokenize
import re
import tensorflow_datasets as tfds
from dotenv import load_dotenv
import os
import helpers
from halo import Halo
import pickle

load_dotenv()
tf.random.set_seed(1234)

BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
BUFFER_SIZE = int(os.environ.get('BUFFER_SIZE'))

datapath = './data/'
filename = 'Conversations_Ben_Phil.csv'
questions, answers = helpers.load_conversations(datapath, filename)

spinner = Halo(text='Creating tokenizer and vocabulary ...',
               spinner='dots')
spinner.start()
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)
VOCAB_SIZE = tokenizer.vocab_size + 2
tokenizer.save_to_file(
    filename_prefix="tokenizer")
spinner.stop()

spinner = Halo(text='Tokenize and filter dataset sentences ...',
               spinner='dots')
spinner.start()
questions, answers = helpers.tokenize_and_filter(questions, answers)

# split into train and test data
spinner.stop()

# decoder inputs use the previous target as input
# remove START_TOKEN from targets
spinner = Halo(text='Create tensors from dataset ...',
               spinner='dots')
spinner.start()
print(answers)
exit()
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))
spinner.stop()

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

tf.data.Dataset.save(dataset, 'data/dataset')
