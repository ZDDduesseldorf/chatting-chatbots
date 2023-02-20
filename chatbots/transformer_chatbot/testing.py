import tensorflow as tf
import tensorflow.python.keras as ks
import data_utils
import sentence_processing
from dotenv import load_dotenv
import os
import transformer
from typing import List
import pandas as pd
ks.backend.clear_session()

VOCAB_SIZE = data_utils.get_vocab_size()
NUM_LAYERS = int(os.environ.get('NUM_LAYERS'))
UNITS = int(os.environ.get('UNITS'))
D_MODEL = int(os.environ.get('D_MODEL'))
NUM_HEADS = int(os.environ.get('NUM_HEADS'))
DROPOUT = float(os.environ.get('DROPOUT'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))
EPOCHS = int(os.environ.get('EPOCHS'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))

load_dotenv()

START_TOKEN, END_TOKEN = data_utils.get_start_and_end_tokens()
tokenizer = data_utils.get_tokenizer()
model = transformer.transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

path = f"./models/{EPOCHS}EPOCHS_{NUM_LAYERS}LAYERS_{NUM_HEADS}HEADS_{UNITS}UNITS_{D_MODEL}DMODEL_{MAX_LENGTH}MAXLENGTH_{BATCH_SIZE}BATCHSIZE/best_model"

model.load_weights(path)


def evaluate(sentence):
    sentence = sentence_processing.preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, axis=0)

    for i in range(MAX_LENGTH):
        predictions = model(
            inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    return predicted_sentence

def testing():
    df = pd.DataFrame(columns=['Questions', 'Answers'])

    #Test questions
    mock_questions= ["How are you?", "What’s up?", "Good morning.", "Tell me something.", "Goodbye.", "How can you help me?", "Happy birthday!", "I have a question.",
     "Do you know a joke?", "Do you love me?", "Will you marry me?", "Do you like people?","Does Santa Claus exist?", "Are you part of the Matrix?", "You’re cute.","Do you have a hobby?", "You’re smart.",
     "Tell me about your personality.", "You’re annoying.", "you suck.", "I want to speak to a human", "Don’t you speak English?!","I want the answer NOW!","Are you a robot?", "What is your name?","How old are you?",
     "What day is it today?","What do you do with my data?","What do you do with my data?","Which languages can you speak?","What is your mother’s name?","Where do you live?","How many people can you speak to at once?",
     "What’s the weather like today?","Are you expensive?","Who’s your boss","Do you like cars?"," Do you get smarter?","It was fun to talk with you!","Which car is your favorite?"
     ]

    for index, question in enumerate(mock_questions):
        df.loc[index] = question, predict(question)

    df.to_csv(f'./testing_results/bot_convo_{EPOCHS}EPOCHS_{NUM_LAYERS}LAYERS_{NUM_HEADS}HEADS_{UNITS}UNITS_{D_MODEL}DMODEL_{MAX_LENGTH}MAXLENGTH_{BATCH_SIZE}BATCHSIZE.csv')

testing()