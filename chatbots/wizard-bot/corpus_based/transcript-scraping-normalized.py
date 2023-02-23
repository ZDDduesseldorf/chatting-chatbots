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

# Load the contents of the YAML file into a dictionary
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Access the values in the configuration
min_df = config["DEFAULT"]["min_df"]
max_df = config["DEFAULT"]["max_df"]
ngram_range = (config["DEFAULT"]["ngram_range_min"], config["DEFAULT"]["ngram_range_max"])

movie_urls = ["Harry_Potter_and_the_Philosopher%27s_Stone",
              "Harry_Potter_and_the_Chamber_of_Secrets",
              "Harry_Potter_and_the_Prisoner_of_Azkaban",
              "Harry_Potter_and_the_Deathly_Hallows_–_Part_2"]

question_and_answers = {}

# load language model
nlp = spacy.load("en_core_web_lg")

# define tfidf
tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range, stop_words={'english'}, sublinear_tf=True)

def extract_speaking_parts(web_data):
    """
    Extracts and cleans up all speaking parts from web data.

    params:
    parts: html
    """     
    content = web_data.find(id="content")
    speaking_parts = []
    for speaker in content.find_all("b"):
        # clean text
        text_cleaned = re.sub("([\(\[]).*?([\)\]])", "", speaker.parent.text)  # remove stage directions
        # Unicode -> Ascii
        #text_cleaned = unicodedata.normalize('NFKD', text_cleaned).encode('ascii','ignore')

        text_cleaned = text_cleaned.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'").replace("\u2018", "'").replace("\u00a0", " ").replace("\u00be", "3/4").replace("\n", "").replace("\u2013", "-")

        speaking_parts.append(text_cleaned)
    assemble_dict(speaking_parts)

# def extract_speaking_parts_billoSeite(web_data):
#     characterName = "snape".upper()
#     regex = "<b> {29}SNAPE\n<\/b>(.|\n)*?<b>"

#     matched = re.findall("<b> {29}SNAPE\n<\/b>(.|\n)*?<b>", web_data)
#     # matched = re.findall("(.|\n)*?", web_data)
#     # matched.split("\n")
#     return matched

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


def assemble_dict(parts):
    """
    This function assembles the dictionary containing lines and the responding lines of spoken text.

    params:
    parts: list of all speaking parts
    """
    
    character_name ="Harry:"

    for line_index, line in enumerate(parts):
        index = line.find(character_name)
        if index == 0:
            previous_line = parts[line_index - 1]
            previous_line_text = previous_line[previous_line.find(":") + 1:]
            question_and_answers[previous_line_text] = line[len(character_name):]

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
    for movie in movie_urls:
        transcript = req.get('https://warnerbros.fandom.com/wiki/' + movie + '/Transcript')
        soup = BeautifulSoup(transcript.text, "html.parser")
        extract_speaking_parts(soup)

    data = pd.DataFrame(question_and_answers.items(), columns=["questions", "responses"])

    # Normalize & Vectorize Data
    question_vectors = vectorize_questions(data)

    # Save
    data.to_json("./data.json")

    # Save the vectors to a file
    with open("vectors.pkl", "wb") as f:
        pickle.dump(question_vectors, f)

    print("Done!")
