import tensorflow as tf
import tensorflow.python.keras as ks
import helpers
from dotenv import load_dotenv
import os
import transformer

ks.backend.clear_session()

VOCAB_SIZE = helpers.get_vocab_size()
NUM_LAYERS = int(os.environ.get('NUM_LAYERS'))
UNITS = int(os.environ.get('UNITS'))
D_MODEL = int(os.environ.get('D_MODEL'))
NUM_HEADS = int(os.environ.get('NUM_HEADS'))
DROPOUT = float(os.environ.get('DROPOUT'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))
EPOCHS = int(os.environ.get('EPOCHS'))

load_dotenv()

START_TOKEN, END_TOKEN = helpers.get_start_and_end_tokens()
tokenizer = helpers.get_tokenizer()
model = transformer.transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

path = f"./models/{EPOCHS}EPOCHS_{MAX_SAMPLES}SAMPLES_{MAX_LENGTH}LENGTH/"

model.load_weights(path)


def evaluate(sentence):
    sentence = helpers.preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

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


while (True):
    inp = input('Say something: ')
    print('Jimmy: ', predict(inp))
