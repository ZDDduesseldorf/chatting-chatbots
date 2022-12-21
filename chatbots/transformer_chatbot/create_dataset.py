from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
from dotenv import load_dotenv
import os
import helpers
from halo import Halo

tf.random.set_seed(1234)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
load_dotenv()

BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
BUFFER_SIZE = int(os.environ.get('BUFFER_SIZE'))
EPOCHS = int(os.environ.get('EPOCHS'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))

datapath = './data/merged/'
path = f"{datapath}{MAX_LENGTH}LENGTH/"
filename = 'merged.csv'
os.mkdir(path)

questions, answers = helpers.load_conversations(datapath, filename)

spinner = Halo(text='Creating tokenizer and vocabulary ...',
               spinner='monkey')
spinner.start()
helpers.create_tokenizer(questions, answers)
spinner.stop()

spinner = Halo(text='Tokenize and filter dataset sentences ...',
               spinner='monkey')
spinner.start()
# split dataset 9:10
val_split = int(len(questions)*(9/10))
train_questions, train_answers = helpers.tokenize_and_filter(
    questions[:val_split], answers[:val_split])
val_questions, val_answers = helpers.tokenize_and_filter(
    questions[val_split:], answers[val_split:])
spinner.stop()

# decoder inputs use the previous target as input
# remove START_TOKEN from targets
spinner = Halo(text='Create tensors from dataset ...',
               spinner='monkey')
spinner.start()
train_dataset = helpers.create_and_save_dataset(
    train_questions, train_answers, "train")
val_dataset = helpers.create_and_save_dataset(
    val_questions, val_answers, "val")
spinner.stop()

print('Size training samples:', len(train_questions))
print('Size val samples:', len(val_questions))
