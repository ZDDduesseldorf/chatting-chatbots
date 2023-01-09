import tensorflow as tf
import transformer
import helpers
import data
import os
from dotenv import load_dotenv
import tensorflow.python.keras as ks
from train_model import CustomSchedule, loss_function

load_dotenv()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

new_params = helpers.Params()
new_loader = data.DataLoader(new_params)

base_params = helpers.Base_Model_Params()
base_loader = data.DataLoader(base_params)
base_tokenizer = base_loader.get_tokenizer()

# new model with new parameters, but old tokenizer and weights
new_model = transformer.transformer(new_params, base_tokenizer.vocab_size)
new_model.load_weights(base_params.weights_path)

learning_rate = CustomSchedule(new_params.d_model)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, new_params.max_length - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


new_model.compile(optimizer=optimizer,
              loss=loss_function, metrics=[accuracy])

# load new datasets
train_dataset, val_dataset = new_loader.get_datasets()

tensorboard_callback = ks.callbacks.TensorBoard(log_dir=new_params.log_dir)

new_model.fit(train_dataset, epochs=10, validation_data=val_dataset, initial_epoch=10)

# save new weights inside base folder under 
new_model.save_weights(f'{base_params.combined_dir}/combined_weights')