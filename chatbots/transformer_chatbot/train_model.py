import helpers
import tensorflow as tf
import transformer
import data
import tensorflow.python.keras as ks

params = helpers.Params()
loader = data.DataLoader(params)

model = transformer.transformer(params, loader.get_vocab_size())

learning_rate = transformer.CustomSchedule(params.d_model)

optimizer = tf.keras.optimizers.Adam(
    learning_rate,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9)


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, params.max_length - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, params.max_length - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=[accuracy])

train_dataset, val_dataset = loader.get_datasets()

helpers.ensure_dir(params.model_dir)
helpers.ensure_dir(params.log_dir)

tensorboard_callback = ks.callbacks.TensorBoard(log_dir=params.log_dir)

checkpoint_callback = ks.callbacks.ModelCheckpoint(
    f"{params.weights_path}_best", save_best_only=True, save_weights_only=True)

stop_early_callback = ks.callbacks.EarlyStopping(
    monitor='val_loss', patience=3)

model.fit(
    train_dataset,
    epochs=params.epochs,
    validation_data=val_dataset,
    callbacks=[tensorboard_callback, checkpoint_callback, stop_early_callback])

print(f"Saving model to path: {params.weights_path}")
model.save_weights(params.weights_path)
