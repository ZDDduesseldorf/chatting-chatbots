import tensorflow as tf
import transformer
import helpers
import os
from dotenv import load_dotenv
import tensorflow.python.keras as ks

load_dotenv()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

""" VOCAB_SIZE = helpers.get_vocab_size()
NUM_LAYERS = int(os.environ.get('NUM_LAYERS'))
UNITS = int(os.environ.get('UNITS'))
D_MODEL = int(os.environ.get('D_MODEL'))
NUM_HEADS = int(os.environ.get('NUM_HEADS'))
DROPOUT = float(os.environ.get('DROPOUT'))
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
EPOCHS = int(os.environ.get('EPOCHS'))

path = f"./models/{EPOCHS}EPOCHS_{MAX_SAMPLES}SAMPLES_{MAX_LENGTH}LENGTH/" """

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

train_dataset = helpers.load_dataset("train")
val_dataset = helpers.load_dataset("val")

logdir =f"{helpers.model_path}\{helpers.EPOCHS}EPOCHS_{helpers.MAX_SAMPLES}SAMPLES_{helpers.MAX_LENGTH}LENGTH"
tensorboard_callback = ks.callbacks.TensorBoard(log_dir=logdir)

model.fit(train_dataset, epochs=helpers.EPOCHS, validation_data=val_dataset)

model.save_weights(f"{helpers.model_path}\{helpers.EPOCHS}EPOCHS_weights")