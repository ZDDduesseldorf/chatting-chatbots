from helpers import Params
import tensorflow as tf
import masks
import positional_encoding
import multi_head_attention


def encoder_layer(params: Params, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, params.d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = multi_head_attention.MultiHeadAttention(
        params.d_model, params.num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    attention = tf.keras.layers.Dropout(rate=params.dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(
        units=params.num_units, activation=params.activation)(attention)
    outputs = tf.keras.layers.Dense(units=params.d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=params.dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(params: Params, vocab_size: int, name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, params.d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(params.d_model, tf.float32))
    embeddings = positional_encoding.PositionalEncoding(
        vocab_size, params.d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=params.dropout)(embeddings)

    for i in range(params.num_layers):
        outputs = encoder_layer(params, name="encoder_layer_{}".format(i))([
            outputs,
            padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(params: Params, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, params.d_model), name="inputs")
    enc_outputs = tf.keras.Input(
        shape=(None, params.d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = multi_head_attention.MultiHeadAttention(
        params.d_model, params.num_heads, name="attention_1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    attention2 = multi_head_attention.MultiHeadAttention(
        params.d_model, params.num_heads, name="attention_2")(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
    attention2 = tf.keras.layers.Dropout(rate=params.dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(
        units=params.num_units, activation=params.activation)(attention2)
    outputs = tf.keras.layers.Dense(units=params.d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=params.dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def decoder(params: Params, vocab_size: int, name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(
        shape=(None, params.d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, params.d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(params.d_model, tf.float32))
    embeddings = positional_encoding.PositionalEncoding(
        vocab_size, params.d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=params.dropout)(embeddings)

    for i in range(params.num_layers):
        outputs = decoder_layer(params, name='decoder_layer_{}'.format(i))(
            inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def transformer(params: Params, vocab_size: int, name="transformer"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        masks.create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        masks.create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        masks.create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(params, vocab_size)(
        inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(params, vocab_size)(
        inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(
        units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
