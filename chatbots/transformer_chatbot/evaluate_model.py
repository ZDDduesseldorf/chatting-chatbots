import tensorflow as tf
import helpers
import data
import transformer

params = helpers.Params()

START_TOKEN, END_TOKEN = data.get_start_and_end_tokens(params)
tokenizer = data.get_tokenizer(params)

model = transformer.transformer(params, data.get_vocab_size(params))

model.load_weights(params.weights_path)


def evaluate(sentence):
    sentence = data.preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for _ in range(params.max_length):
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
    inp = input('>> ')
    if inp in ["q", "quit", "exit"]:
        break
    print(predict(inp))
