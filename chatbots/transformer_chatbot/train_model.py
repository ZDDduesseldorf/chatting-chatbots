import tensorflow as tf
import transformer
import helpers
import os
from dotenv import load_dotenv
import tensorflow.python.keras as ks

load_dotenv()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

VOCAB_SIZE = helpers.get_vocab_size()
NUM_LAYERS = int(os.environ.get('NUM_LAYERS'))
UNITS = int(os.environ.get('UNITS'))
D_MODEL = int(os.environ.get('D_MODEL'))
NUM_HEADS = int(os.environ.get('NUM_HEADS'))
DROPOUT = float(os.environ.get('DROPOUT'))
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
EPOCHS = int(os.environ.get('EPOCHS'))

path = f"./models/{EPOCHS}EPOCHS_{MAX_SAMPLES}SAMPLES_{MAX_LENGTH}LENGTH/"

model = transformer.transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

learning_rate = transformer.CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


model.compile(optimizer=optimizer,
              loss=transformer.loss_function, metrics=['accuracy'])

train_dataset = helpers.load_dataset("train")
val_dataset = helpers.load_dataset("val")

# Only for pretrained models!
# path_pretrained_model = "./models/10EPOCHS_0SAMPLES_10LENGTH/best_model"
# model = model.load_weights(path_pretrained_model)

path_pretrained = "./models/10EPOCHS_0SAMPLES_10LENGTH/"

model.load_weights(path_pretrained)

logdir =f"logs/scalars/{EPOCHS}EPOCHS_{MAX_SAMPLES}SAMPLES_{MAX_LENGTH}LENGTH"
tensorboard_callback = ks.callbacks.TensorBoard(log_dir=logdir)
checkpoint_callback = ks.callbacks.ModelCheckpoint(f"{path}best_model", save_best_only=True, save_weights_only= True)
stop_early_callback = ks.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[tensorboard_callback, stop_early_callback])

model.save_weights(path)