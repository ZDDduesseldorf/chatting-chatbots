from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()


DATASET_NAME = os.environ.get('DATASET')
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
TARGET_VOCAB_SIZE = 2**int(os.environ.get('VOCAB_SIZE_EXPONENT'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
BUFFER_SIZE = int(os.environ.get('BUFFER_SIZE'))
NUM_LAYERS = int(os.environ.get('NUM_LAYERS'))
NUM_HEADS = int(os.environ.get('NUM_HEADS'))
EPOCHS = int(os.environ.get('EPOCHS'))
D_MODEL = int(os.environ.get('D_MODEL'))
UNITS = int(os.environ.get('UNITS'))
DROPOUT = float(os.environ.get('DROPOUT'))

with open('.env', 'r') as f:
    print(f.read())


TOKENIZER_KEY = f"{DATASET_NAME}/{TARGET_VOCAB_SIZE}Voc"
DATASET_KEY = f"{TOKENIZER_KEY}/{MAX_SAMPLES}Smp_{MAX_LENGTH}Len_{BATCH_SIZE}Bat_{BUFFER_SIZE}Buf"
MODEL_KEY = f"{DATASET_KEY}/{NUM_LAYERS}Lay_{NUM_HEADS}Hed_{EPOCHS}Epo"
LOGS_KEY = MODEL_KEY.replace('/', '__')

DATA_DIR = "data"
TOKENIZER_DIR = f"{DATA_DIR}/{TOKENIZER_KEY}"
TOKENIZER_PATH = f"{TOKENIZER_DIR}/tokenizer"
DATASET_DIR = f"{DATA_DIR}/{DATASET_KEY}"
MODEL_DIR = f"{DATA_DIR}/{MODEL_KEY}"
WEIGHTS_PATH = f"{MODEL_DIR}/weights"

LOGS_DIR = f"logs/{LOGS_KEY}"


def ensure_dir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)
