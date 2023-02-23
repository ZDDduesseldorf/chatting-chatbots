#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import re
import requests as req
import spacy
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import pickle
import yaml
import json
import os
import copy

# Load the contents of the YAML file into a dictionary
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Access the values in the configuration
min_df = config["TFIDF"]["min_df"]
max_df = config["TFIDF"]["max_df"]
ngram_range = (config["TFIDF"]["ngram_range_min"], config["TFIDF"]["ngram_range_max"])

# load language model
nlp = spacy.load("en_core_web_lg")

# define tfidf
tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range, stop_words={'english'}, sublinear_tf=True)


def vectorize_questions(data):
    """
    This function normalizes the data by applying a Tfidf-Vectorizer.

    returns:
    question_vectors: question vectors
    """
    tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer, ngram_range= (1, 1))
    
    # Tfidf
    tfidf.fit_transform(data.questions + data.responses)

    question_vectors = tfidf.transform(data.questions)

    return question_vectors


def assemble_dict(dialogues): 
    questions = copy.deepcopy(dialogues)
    responses = copy.deepcopy(dialogues)

    questions.pop()
    responses.pop(0)

    data = {'questions': questions, 'responses': responses}
    return pd.DataFrame(data, columns=["questions", "responses"])

def load_json_files(directory):
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                json_files.append(json.load(file))
    return json_files

def combine_json_chunks(json_files):
    all_dialogues = []
    for json_file in json_files:
        dialogues = [item["dialog"] for item in json_file]
        all_dialogues.extend(dialogues)

    return all_dialogues


def spacy_tokenizer(sentence):
    from spacy.lang.en import English
    import string

    parser = English()
    punctuations = string.punctuation
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens

# MAIN METHOD
if __name__ == '__main__':
    # Load Data
    # requesting website

    json_files = load_json_files("corpus_based/daily-dialogue-data")

    dialogues = combine_json_chunks(json_files)

    data = assemble_dict(dialogues)

    # Normalize & Vectorize Data
    question_vectors = vectorize_questions(data)

    # Save
    data.to_json("./corpus_based/data_daily_dialogue.json")

    # Save the vectors to a file
    with open("./corpus_based/vectors_daily_dialogue.pkl", "wb") as f:
        pickle.dump(question_vectors, f)

    print("Done!")
