import json
import os.path

import glob
import re
import zipfile
from enum import Enum
from io import BytesIO
from typing import List

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


class CorpusType(Enum):
    GoT = 'got'  # got transcripts
    Cornell = 'cornell'  # cornell movie dialogs
    Parliament = 'parliament'  # Parliament Question Time Corpus
    DailyDialogs = 'dailyDialogs'  # Daily Dialogs Corpus


def files_in_directory(directory_path, file_patterns=None, recursive=False):
    if file_patterns is None:
        file_patterns = ['**']
    elif not isinstance(file_patterns, list):
        file_patterns = [file_patterns]

    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(os.path.join(directory_path, pattern), recursive=recursive))

    return files


def read_textfile(textfile, mode='lines', encoding='utf-8'):
    """
    Reads a textfile.
    :param textfile: The textfile path.
    :param mode: Determines the return type. 'lines' for a list of textfile lines or 'text' for one string containing
    all file content.
    :param encoding: The encoding of the textfile.
    :return: The content of the textfile.
    """
    f = open(textfile, 'r', encoding=encoding)
    if mode == 'lines':
        text = f.readlines()
    elif mode == 'text':
        text = f.read()
    else:
        raise NotImplementedError('The given mode {} is not implemented!'.format(mode))
    f.close()

    return text


def read_jsonfile(jsonfile):
    """
    Reads a jsonfile.
    :param jsonfile: The jsonfile path.
    :return: The object.
    """
    f = open(jsonfile, 'r')
    json_object = json.load(f)
    f.close()

    return json_object


def download_and_extract_zip_from_url(url, save_path):
    os.makedirs(save_path, exist_ok=True)
    url_request = requests.get(url, stream=True)
    zip_file = zipfile.ZipFile(BytesIO(url_request.content))
    zip_file.extractall(path=save_path)


def get_word_blacklist_regex(blacklist_file):
    word_blacklist = read_textfile(blacklist_file)
    word_blacklist = [line[:-1].lower() for line in word_blacklist]
    word_blacklist = set(word_blacklist)
    word_regex = '|'.join(word_blacklist)
    regex = r'(?i)\b({})\b'.format(word_regex)
    regex = re.compile(regex)

    return regex


def get_available_corpora(data_path, csv_name_format, request_vectors_name_format):
    available_corpora = []
    for corpus_type in CorpusType:
        csv_name = csv_name_format.format(corpus_type.value)
        request_vectors_name = request_vectors_name_format.format(corpus_type.value)
        csv_path = os.path.join(data_path, 'Corpora', csv_name)
        request_vectors_path = os.path.join(data_path, 'Corpora', request_vectors_name)

        if os.path.isfile(csv_path) and os.path.isfile(request_vectors_path):
            available_corpora.append({'csv_path': csv_path, 'request_vectors_path': request_vectors_path})

    return available_corpora


def load_corpora_csvs(corpora_types: List[CorpusType]):
    # load corpora
    rr_pairs = pd.DataFrame()
    for corpus in tqdm(corpora_types, unit='Corpora', desc='Load Corpora'):
        rr = pd.read_csv(corpus['csv_path'], encoding='utf-8')
        rr_pairs = pd.concat([rr_pairs, rr.astype(str)], ignore_index=True)

    return rr_pairs


def load_spacy_vectors(corpora_types: List[CorpusType]):
    # load spacy request vectors
    spacy_request_vectors = np.empty((0, 300), dtype=float)
    for corpus in tqdm(corpora_types, unit='Corpora', desc='Load Spacy Vectors'):
        vectors = np.load(corpus['request_vectors_path'])
        spacy_request_vectors = np.concatenate([spacy_request_vectors, vectors], axis=0)

    return spacy_request_vectors


def multi_replace(text, replacement_dict):
    if len(replacement_dict) == 0:
        return text

    pattern = re.compile("|".join(replacement_dict.keys()))
    return pattern.sub(lambda m: replacement_dict[re.escape(m.group(0))], text)
