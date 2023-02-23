import argparse
from pathlib import Path
import os
from os.path import abspath
import dotenv

dotenv.load_dotenv()


def env(name: str, default: str):
    return os.environ.get(name, default)


def ensure_dir(dir: str):
    Path(dir).mkdir(parents=True, exist_ok=True)


class Params:
    def __init__(self, ignore_shell_args=False, training_stage=0):
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

        parser.add_argument(
            "--stage", default=int(env("STAGE", "0")), type=int)
        parser.add_argument(
            "--super-tokenizer", default=0, type=int)

        if ignore_shell_args:
            args = parser.parse_args([])
        else:
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

        if training_stage < 0:
            self.stage: int = args.stage
        else:
            self.stage = training_stage

        tokenizer_key = f"{self.dataset}/{self.target_vocab_size}Voc"
        dataset_key = f"{tokenizer_key}/{self.max_samples}Smp_{self.max_length}Len_{self.batch_size}Bat_{self.buffer_size}Buf"
        if self.stage == 0:
            model_key = f"{dataset_key}/{self.num_layers}Lay_{self.num_heads}Hed_{self.epochs}Epo"
        else:
            model_key = f"stage{self.stage}/{dataset_key}/{self.num_layers}Lay_{self.num_heads}Hed_{self.epochs}Epo"

        self.tokenizer_dir = abspath(f"{self.data_root}/{tokenizer_key}")
        self.tokenizer_path = abspath(f"{self.tokenizer_dir}/tokenizer")
        self.dataset_dir = abspath(f"{self.data_root}/{dataset_key}")
        self.model_dir = abspath(f"{self.data_root}/{model_key}")
        self.best_weights_path = abspath(f"{self.model_dir}/weights_best")
        self.weights_path = abspath(f"{self.model_dir}/weights")
        self.log_dir = abspath(f"logs/{model_key.replace('/', '__')}")

        d = self.__dict__
        if ignore_shell_args:
            print(
                '\033[93m Ignoring shell-Arguments for these model parameters: \033[0m')
        else:
            print('\033[93m Model Parameters: \033[0m')
        for k in d.keys():
            print(f"{k}: {d[k]}")
        print()
