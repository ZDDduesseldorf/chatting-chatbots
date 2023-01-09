import argparse
from pathlib import Path
import os
from os.path import abspath
import dotenv
import sys

dotenv.load_dotenv()


def env(name: str, default: str):
    return os.environ.get(name, default)


def ensure_dir(dir: str):
    Path(dir).mkdir(parents=True, exist_ok=True)


class Params:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--data-root", default="data", type=str)

        parser.add_argument(
            "--dataset", default=env("DATASET_NAME", "merged"), type=str)
        parser.add_argument(
            "--target-vocab-size-exp",
            default=int(env("TARGET_VOCAB_SIZE_EXP", "14")),
            type=int)
        parser.add_argument(
            "--max-samples", default=int(env("MAX_SAMPLES", "0")), type=int)
        parser.add_argument(
            "--max-length", default=int(env("MAX_LENGTH", "25")), type=int)
        parser.add_argument(
            "--buffer-size", default=int(env("BUFFER_SIZE", "1000000")), type=int)
        parser.add_argument(
            "--batch-size", default=int(env("BATCH_SIZE", "512")), type=int)
        parser.add_argument(
            "--epochs", default=int(env("EPOCHS", "20")), type=int)

        parser.add_argument(
            "--num-layers", default=int(env("NUM_LAYERS", "2")), type=int)
        parser.add_argument(
            "--num-units", default=int(env("UNITS", "512")), type=int)
        parser.add_argument(
            "--d-model", default=int(env("D_MODEL", "256")), type=int)
        parser.add_argument(
            "--num-heads", default=int(env("NUM_HEADS", "8")), type=int)
        parser.add_argument(
            "--dropout", default=float(env("DROPOUT", "0.1")), type=float)
        parser.add_argument("--activation", default="relu", type=str)

        args = parser.parse_args()

        self.data_root: str = abspath(args.data_root)
        self.dataset: str = args.dataset
        self.target_vocab_size: int = 2 ** args.target_vocab_size_exp
        self.max_samples: int = args.max_samples
        self.max_length: int = args.max_length
        self.buffer_size: int = args.buffer_size
        self.batch_size: int = args.batch_size
        self.epochs: int = args.epochs

        self.num_layers: int = args.num_layers
        self.num_units: int = args.num_units
        self.d_model: int = args.d_model
        self.num_heads: int = args.num_heads
        self.dropout: float = args.dropout
        self.activation: str = args.activation

        tokenizer_key = f"{self.dataset}/{self.target_vocab_size}Voc"
        dataset_key = f"{tokenizer_key}/{self.max_samples}Smp_{self.max_length}Len_{self.batch_size}Bat_{self.buffer_size}Buf"
        model_key = f"{dataset_key}/{self.num_layers}Lay_{self.num_heads}Hed_{self.epochs}Epo"

        self.tokenizer_dir = abspath(f"{self.data_root}/{tokenizer_key}")
        self.tokenizer_path = abspath(f"{self.tokenizer_dir}/tokenizer")
        self.dataset_dir = abspath(f"{self.data_root}/{dataset_key}")
        self.model_dir = abspath(f"{self.data_root}/{model_key}")
        self.weights_path = abspath(f"{self.model_dir}/weights")
        self.log_dir = abspath(f"logs/{model_key.replace('/', '__')}")

        d = self.__dict__
        print('\033[93m Model Parameters: \033[0m')
        for k in d.keys():
            print(f"{k}: {d[k]}")
        print()


class Base_Model_Params:
    def __init__(self):
        self.data_root: str = abspath("data")
        self.dataset: str = env("DATASET_NAME", "merged")
        self.target_vocab_size: int = 2 ** int(env("TARGET_VOCAB_SIZE_EXP", "14"))
        self.max_samples: int = int(env("MAX_SAMPLES", "0"))
        self.max_length: int = int(env("MAX_LENGTH", "25"))
        self.buffer_size: int = int(env("BUFFER_SIZE", "1000000"))
        self.batch_size: int = int(env("BATCH_SIZE", "512"))
        self.epochs: int = int(env("EPOCHS", "20"))

        self.num_layers: int = int(env("NUM_LAYERS", "2"))
        self.num_units: int = int(env("UNITS", "512"))
        self.d_model: int = int(env("D_MODEL", "256"))
        self.num_heads: int = int(env("NUM_HEADS", "8"))
        self.dropout: float = float(env("DROPOUT", "0.1"))
        self.activation: str = "relu"

        tokenizer_key = f"{self.dataset}/{self.target_vocab_size}Voc"
        dataset_key = f"{tokenizer_key}/{self.max_samples}Smp_{self.max_length}Len_{self.batch_size}Bat_{self.buffer_size}Buf"
        model_key = f"{dataset_key}/{self.num_layers}Lay_{self.num_heads}Hed_{self.epochs}Epo"

        self.tokenizer_dir = abspath(f"{self.data_root}/{tokenizer_key}")
        self.tokenizer_path = abspath(f"{self.tokenizer_dir}/tokenizer")
        self.dataset_dir = abspath(f"{self.data_root}/{dataset_key}")
        self.model_dir = abspath(f"{self.data_root}/{model_key}")
        self.weights_path = abspath(f"{self.model_dir}/weights")
        self.log_dir = abspath(f"logs/{model_key.replace('/', '__')}")

        args_string = ''
        for i in range(1, len(sys.argv)):
            args_string += sys.argv[i] + '+'

        self.combined_dir = abspath(f"{self.model_dir}/{args_string}")
        ensure_dir(self.combined_dir)

        d = self.__dict__
        print('Continuing Training based on the model specified in .env file.')
        print(f'Saving combined model to: {self.combined_dir}')
        print('\033[93m Base model Parameters: \033[0m')
        for k in d.keys():
            print(f"{k}: {d[k]}")
        print()