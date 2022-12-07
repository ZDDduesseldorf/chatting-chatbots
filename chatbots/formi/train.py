import os
import tensorflow as tf
from lib.data import load_dataset, MAX_UTTERANCE_LENGTH
from lib.model import create_model, D_MODEL

tf.keras.backend.clear_session()

dataset = load_dataset()
model = create_model()


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_UTTERANCE_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9,
)

model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=["accuracy"],
)

# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#     log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
#     histogram_freq=1,
# )

model.fit(
    dataset,
    epochs=int(os.getenv("EPOCHS", "5")),
    # callbacks=[tensorboard_callback],
)

model.save_weights("weights/weights")
