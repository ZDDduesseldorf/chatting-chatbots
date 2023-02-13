import tensorflow as tf
from dotenv import load_dotenv
import os
import data_utils
from halo import Halo
load_dotenv()
tf.random.set_seed(1234)

# check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
BUFFER_SIZE = int(os.environ.get('BUFFER_SIZE'))
EPOCHS = int(os.environ.get('EPOCHS'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))

spinner = Halo(text='Creating tokenizer and vocabulary ...',
               spinner='monkey')
spinner.start()

# load conversation from csv containing 2M question/answer samples
datapath = './data/merged/'
path = f"{datapath}{MAX_LENGTH}LENGTH/"
filename = 'merged.csv'
os.mkdir(path)
questions, answers = data_utils.load_conversations(datapath, filename)

# create and persist tokenizer
data_utils.create_tokenizer(questions, answers)

# split dataset into 90% train and 10% validation data
val_split = int(len(questions)*(9/10))
train_questions, train_answers = data_utils.tokenize_and_filter(
    questions[:val_split], answers[:val_split])
val_questions, val_answers = data_utils.tokenize_and_filter(
    questions[val_split:], answers[val_split:])

# persist training and validation dataset
train_dataset = data_utils.create_and_save_dataset(
    train_questions, train_answers, "train")
val_dataset = data_utils.create_and_save_dataset(
    val_questions, val_answers, "val")

spinner.stop()

print('Size training samples:', len(train_questions))
print('Size val samples:', len(val_questions))



