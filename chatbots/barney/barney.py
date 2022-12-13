import os
import config
import csv
from config import csv_quotechar, csv_separator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas


corpus = {}
ignored_episodes = []

for file_name in os.listdir(config.processed_resources_folder_name):
    path = os.path.join(config.processed_resources_folder_name, file_name)
    
    if path.find("csv") == -1:
        continue

    # check if file is to small
    with open(path, "r") as file:
        lines = file.readlines()
        if len(lines) < 15:
            ignored_episodes.append(file_name)
            continue

    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=csv_separator, quotechar=csv_quotechar)
        
        for prior_message, barney_message in reader:
            # skip header
            if reader.line_num == 1:
                continue
            corpus[prior_message] = barney_message

print(f"Following episodes got ignored {ignored_episodes}")
print(f"{len(ignored_episodes)} of 208 episodes got ignored. That means {round(len(ignored_episodes) / 208 * 100)}% got ignored")
print(f"The corpus has {len(corpus)} pairs")

print("first 5 pairs are:") 
for index, (key, value) in enumerate(corpus.items()):
    if index == 5:
        break
    print(f"prior message: {key}" )
    print(f"barney's message: {value}")


tfidf = TfidfVectorizer(min_df=2, max_df = 0.5, ngram_range=(1, 1))
features = tfidf.fit_transform(corpus.keys())
print(features.shape)


def respond(input):
    input_tfidf = tfidf.transform([input])
    similarities = cosine_similarity(input_tfidf, features)
    idx = np.argsort(similarities)[0][-1]
    print(idx)

    return

print(respond("test"))