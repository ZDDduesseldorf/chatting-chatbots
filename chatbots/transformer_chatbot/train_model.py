import tensorflow as tf
import transformer
import data_utils
import os
from dotenv import load_dotenv
import tensorflow.python.keras as ks

load_dotenv()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#load hyperparamters from .env
VOCAB_SIZE = data_utils.get_vocab_size()
NUM_LAYERS = int(os.environ.get('NUM_LAYERS'))
UNITS = int(os.environ.get('UNITS'))
D_MODEL = int(os.environ.get('D_MODEL'))
NUM_HEADS = int(os.environ.get('NUM_HEADS'))
DROPOUT = float(os.environ.get('DROPOUT'))
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
EPOCHS = int(os.environ.get('EPOCHS'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))

path = f"./models/{EPOCHS}EPOCHS_{NUM_LAYERS}LAYERS_{NUM_HEADS}HEADS_{UNITS}UNITS_{D_MODEL}DMODEL_{MAX_LENGTH}MAXLENGTH_{BATCH_SIZE}BATCHSIZE/"

#Set model architecture
model = transformer.transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

#Set learning rate and optimzer (Adam)
learning_rate = transformer.CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer,
              loss=transformer.loss_function, metrics=[transformer.accuracy])

#Load training and validation dataset
train_dataset = data_utils.load_dataset("train")
val_dataset = data_utils.load_dataset("val")

#Create callbacks for early stopping, checkpointing and tensorboard logs
logdir = f"logs/scalars/{EPOCHS}EPOCHS_{NUM_LAYERS}LAYERS_{NUM_HEADS}HEADS_{UNITS}UNITS_{D_MODEL}DMODEL_{MAX_LENGTH}MAXLENGTH_{BATCH_SIZE}BATCHSIZE"
tensorboard_callback = ks.callbacks.TensorBoard(log_dir=logdir)
checkpoint_callback = ks.callbacks.ModelCheckpoint(
    f"{path}best_model", save_best_only=True, save_weights_only=True)

# Implement early stopping in case loss does not decrease for 3 epochs
stop_early_callback = ks.callbacks.EarlyStopping(
    monitor='val_loss', patience=3)
    
# Init model training with stop early function and save the weights
model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset,
          callbacks=[tensorboard_callback, checkpoint_callback, stop_early_callback])

# Save weights for evaluation
model.save_weights(f"{path}final_model")
