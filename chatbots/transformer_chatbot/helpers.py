import argparse
from pathlib import Path


def ensure_dir(dir: str):
    Path(dir).mkdir(parents=True, exist_ok=True)


class Params:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--data-root", default="data", type=str)

        parser.add_argument("--dataset", default="merged", type=str)
        parser.add_argument("--target-vocab-size-exp", default=14, type=int)
        parser.add_argument("--max-samples", default=0, type=int)
        parser.add_argument("--max-length", default=30, type=int)
        parser.add_argument("--buffer-size", default=20000, type=int)
        parser.add_argument("--batch-size", default=64, type=int)
        parser.add_argument("--epochs", default=15, type=int)

        parser.add_argument("--num-layers", default=2, type=int)
        parser.add_argument("--num-units", default=512, type=int)
        parser.add_argument("--d-model", default=256, type=int)
        parser.add_argument("--num-heads", default=8, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--activation", default="relu", type=str)

        args = parser.parse_args()

        for k in vars(args).keys():
            print(f"{k}: {vars(args)[k]}")

        self.data_root: str = args.data_root
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

        self.tokenizer_dir = f"{self.data_root}/{tokenizer_key}"
        self.tokenizer_path = f"{self.tokenizer_dir}/tokenizer"
        self.dataset_dir = f"{self.data_root}/{dataset_key}"
        self.model_dir = f"{self.data_root}/{model_key}"
        self.weights_path = f"{self.model_dir}/weights"
        self.log_dir = f"logs/{model_key.replace('/', '__')}"
