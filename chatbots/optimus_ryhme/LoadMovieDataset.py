import os
import re
from tqdm import tqdm
import tensorflow as tf
tf.random.set_seed(1234)
from PreprocessSentence import preprocess_sentence

# ------------- Directory Paths ----------------------
path_to_zip = tf.keras.utils.get_file('cornell_movie_dialogs.zip', origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip', extract=True)

path_to_dataset = os.path.join(os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")
path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt')

datasetPath = "MovieDataSet"

def getDatasetDirectory():
    return datasetPath

def load_conversations(MAX_SAMPLES):
    # dictionary of line id to text
    id2line = {}

    with open(path_to_movie_lines, errors='ignore') as file:
        lines = file.readlines()

    for line in tqdm(lines, desc = "Splitting lines: "):
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []

    with open(path_to_movie_conversations, 'r') as file:
        lines = file.readlines()

    for line in tqdm(lines, desc = "Preprocessing sentences: "):
        parts = line.replace('\n', '').split(' +++$+++ ')
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]

        for i in range(len(conversation) - 1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))

            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs

    return inputs, outputs