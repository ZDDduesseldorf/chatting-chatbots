import os
import config
import csv
from config import csv_quotechar, csv_separator
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


corpus = {}

for file_name in os.listdir(config.processed_resources_folder_name):
    path = os.path.join(config.processed_resources_folder_name, file_name)
    
    if path.find("csv") == -1:
        continue

    # check if file is to small
    with open(path, "r") as file:
        lines = file.readlines()
        if len(lines) < 15:
            continue

    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=csv_separator, quotechar=csv_quotechar)
        
        for prior_message, barneys_message in reader:
            # skip header
            if reader.line_num == 1:
                continue
            corpus[prior_message] = barneys_message

def respond(input):
    corpus_df = pd.DataFrame({"prior_message": list(corpus.keys()), "barneys_message": list(corpus.values())})

    tfidf = TfidfVectorizer(min_df=2, max_df = 0.5, ngram_range=(1, 2))
    prior_messages_tfidf = tfidf.fit_transform(corpus_df.prior_message)
    input_tfidf = tfidf.transform([input])

    similarities = cosine_similarity(input_tfidf, prior_messages_tfidf)
    idx = np.argsort(similarities)[0][-1]
    sorted_values = np.sort(similarities)
    print(len(tfidf.get_feature_names_out()))
    return corpus_df.loc[idx, "barneys_message"]