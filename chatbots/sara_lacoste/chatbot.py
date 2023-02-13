import tensorflow as tf
import data
from lib.tokenizer import TransformerTokenizer


class TransformerChatbot:
    def __init__(self, model: tf.keras.Model, tokenizer: TransformerTokenizer, max_length: int):
        self._model = model
        self._tokenizer = tokenizer
        self._max_length = max_length

    def _evaluate(self, sentence: str):
        sentence = data.preprocess_sentence(sentence)

        START, END = [self._tokenizer.start_token], [self._tokenizer.end_token]
        sentence = tf.expand_dims(
            START + self._tokenizer.encode(sentence) + END, axis=0)

        output = tf.expand_dims(START, 0)

        for _ in range(self._max_length):
            predictions = self._model(
                inputs=[sentence, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self._tokenizer.end_token):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)

    def predict(self, sentence: str):
        prediction = self._evaluate(sentence)
        predicted_sentence = self._tokenizer.decode(
            [i for i in prediction if i < self._tokenizer.corpus_vocab_size])
        return predicted_sentence
