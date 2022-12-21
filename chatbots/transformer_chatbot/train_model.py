import tensorflow as tf
import transformer
import helpers
import data
import tensorflow.python.keras as ks

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if len(helpers.WEIGHTS_PATH) >= 91:
    raise ValueError(
        f"Weights output path too long (length: {len(helpers.WEIGHTS_PATH)}>=91): {helpers.WEIGHTS_PATH}")

tf.keras.backend.clear_session()

model = transformer.transformer(
    vocab_size=data.get_vocab_size(),
    num_layers=helpers.NUM_LAYERS,
    units=helpers.UNITS,
    d_model=helpers.D_MODEL,
    num_heads=helpers.NUM_HEADS,
    dropout=helpers.DROPOUT)

learning_rate = transformer.CustomSchedule(helpers.D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9)


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, helpers.MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


model.compile(
    optimizer=optimizer,
    loss=transformer.loss_function,
    metrics=[accuracy])

train_dataset, val_dataset = data.get_datasets()

helpers.ensure_dir(helpers.LOGS_DIR)
tensorboard_callback = ks.callbacks.TensorBoard(log_dir=helpers.LOGS_DIR)

checkpoint_callback = ks.callbacks.ModelCheckpoint(
    f"{helpers.WEIGHTS_PATH}_best", save_best_only=True, save_weights_only=True)

stop_early_callback = ks.callbacks.EarlyStopping(
    monitor='val_loss', patience=3)


model.fit(
    train_dataset,
    epochs=helpers.EPOCHS,
    validation_data=val_dataset,
    callbacks=[tensorboard_callback, checkpoint_callback, stop_early_callback])

print(f"Saving model to path: {helpers.WEIGHTS_PATH}")

helpers.ensure_dir(helpers.MODEL_DIR)

model.save_weights(helpers.WEIGHTS_PATH)
