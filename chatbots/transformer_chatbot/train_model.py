import tensorflow as tf
import transformer
import helpers
from dotenv import load_dotenv
import tensorflow.python.keras as ks

load_dotenv()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.keras.backend.clear_session()

model = transformer.transformer(
    vocab_size=helpers.VOCAB_SIZE,
    num_layers=helpers.NUM_LAYERS,
    units=helpers.UNITS,
    d_model=helpers.D_MODEL,
    num_heads=helpers.NUM_HEADS,
    dropout=helpers.DROPOUT)

learning_rate = transformer.CustomSchedule(helpers.D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


model.compile(optimizer=optimizer, loss=transformer.loss_function)


if helpers.checkAndCreateDirectory(helpers.model_path):
    train_dataset = helpers.load_dataset("train")
    val_dataset = helpers.load_dataset("val")
else:
    train_dataset, val_dataset = helpers.create_dataset()


logdir =f"{helpers.model_path}/{helpers.EPOCHS}EPOCHS_{helpers.MAX_SAMPLES}SAMPLES_{helpers.MAX_LENGTH}LENGTH"
helpers.checkAndCreateDirectory(logdir)
tensorboard_callback = ks.callbacks.TensorBoard(log_dir=logdir)

model.fit(train_dataset, epochs=helpers.EPOCHS, validation_data=val_dataset)

abs_path = f"{helpers.model_path}/{helpers.EPOCHS}Epo_wghts"

print(f"Saving model to path: {abs_path}")
print(f"The path length needs to be somewhere below 91 or model.save_weights will fail! path lenght: {len(abs_path)}")

model.save_weights(abs_path)