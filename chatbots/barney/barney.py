import csv
import os
import re
from typing import Dict

import config
import numpy as np
import pandas as pd
import spacy
from config import csv_quotechar, csv_separator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus: Dict[str,str] = {}

for file_name in os.listdir(config.processed_resources_folder_name):
    path = os.path.join(config.processed_resources_folder_name, file_name)
    
    if path.find("csv") == -1:
        continue

    # check if file is to small
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        if len(lines) < 15:
            continue

    with open(path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=csv_separator, quotechar=csv_quotechar)
        
        for prior_message, barneys_message in reader:
            # skip header
            if reader.line_num == 1:
                continue
            corpus[prior_message] = barneys_message


def replace_entity(message: str, new_entity: str) -> str:
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("merge_entities")
    doc = nlp(message)
    
    response = message
    ent_to_replace = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            ent_to_replace = ent
            break

    if(ent_to_replace):
        response = re.sub(ent_to_replace.text, new_entity, message)
    return response

def respond(user_input: str) -> str:
    corpus_df = pd.DataFrame({"prior_message": list(corpus.keys()), "barneys_message": list(corpus.values())})

    tfidf = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range = (1, 2))
    prior_messages_tfidf = tfidf.fit_transform(corpus_df.prior_message)
    input_tfidf = tfidf.transform([user_input])

    similarities = cosine_similarity(input_tfidf, prior_messages_tfidf)
    idx = np.argsort(similarities)[0][-1]
    return corpus_df.loc[idx, "barneys_message"]
    