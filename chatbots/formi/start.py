import tensorflow as tf
from lib.data import DataController, preprocess_utterance, MAX_UTTERANCE_LENGTH
from lib.model import create_model


ctrl = DataController()
start_token, end_token = ctrl.get_boundary_tokens()
tokenizer = ctrl.get_tokenizer()

model = create_model()
model.load_weights("weights/weights")


def evaluate(sentence):
    sentence = preprocess_utterance(sentence)

    sentence = tf.expand_dims(
        [start_token] + tokenizer.encode(sentence) + [end_token],
        axis=0,
    )

    output = tf.expand_dims([start_token], 0)

    for _ in range(MAX_UTTERANCE_LENGTH):
        predictions = model(
            inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, end_token):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )

    return predicted_sentence


while (True):
    inp = input("> ")
    if inp in ["q", "quit", "exit"]:
        break
    # print("Output: ", predict(inp))
    print(predict(inp))
