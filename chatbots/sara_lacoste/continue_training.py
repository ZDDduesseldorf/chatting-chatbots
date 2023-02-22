import tensorflow as tf
import pandas as pd
import tensorflow.python.keras as ks
from dotenv import load_dotenv
from lib.helpers import Params, ensure_dir
from lib.data import DataLoader
from lib.transformer import create_model
from train_model import CustomSchedule, loss_function

load_dotenv()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

base_params = Params(ignore_shell_args=True)
base_loader = DataLoader(base_params)
super_tokenizer = base_loader.get_tokenizer()

stage = base_params.stage + 1
new_params = Params(training_stage=stage)
new_loader = DataLoader(new_params)

print(f'\033[93m Base Params weights path: {base_params.weights_path} \033[0m')
# new model with new parameters, but old tokenizer and weights
new_model = create_model(new_params, super_tokenizer.vocab_size)
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

new_model.fit(train_dataset, epochs=10,
              validation_data=val_dataset, initial_epoch=10)

ensure_dir(new_params.model_dir)
ensure_dir(new_params.log_dir)

# save training progression to csv
base_log = pd.read_csv(f"{base_params.log_dir}/training_params.csv")
new_log = base_log.append(pd.DataFrame([vars(new_params)]))
new_log.to_csv(f"{new_params.log_dir}/training_params.csv")

# save new weights inside base folder under
new_model.save_weights(new_params.weights_path)
