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

loadWeights = helpers.checkAndCreateDirectory(helpers.model_path)
loadTokenizer = os.path.exists(f"{helpers.tokenizer_path}.subwords")


questions, answers = helpers.load_conversations()

if loadTokenizer:
    spinner = Halo(text=f"Loading Tokenizer from {helpers.tokenizer_path}", spinner='dots')
    spinner.start()
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(helpers.tokenizer_path)
else:
    # Build tokenizer using tfds for both questions and answers
    spinner = Halo(text='Creating tokenizer and vocabulary ...', spinner='dots')
    spinner.start()
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=helpers.VOCAB_SIZE)
    tokenizer.save_to_file(helpers.tokenizer_path)

spinner.stop()


spinner = Halo(text='Tokenize and filter dataset sentences ...', spinner='dots')
spinner.start()

#split dataset 70:30
val_split=int(len(questions)*(2/3))
train_questions, train_answers = helpers.tokenize_and_filter(questions[:val_split], answers[:val_split], tokenizer)
val_questions, val_answers = helpers.tokenize_and_filter(questions[val_split:], answers[val_split:], tokenizer)

# split into train and test data
spinner.stop()

# decoder inputs use the previous target as input
# remove START_TOKEN from targets
spinner = Halo(text='Create tensors from dataset ...', spinner='dots')
spinner.start()
train_dataset = helpers.create_and_save_dataset(train_questions, train_answers, "train")
val_dataset = helpers.create_and_save_dataset(val_questions, val_answers, "val")
spinner.stop()

print('Size training samples:', len(train_questions))
print('Size val samples:', len(val_questions))

