import os
import config
import csv
from config import csv_quotechar, csv_separator
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


questions_and_answers = {}
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
            questions_and_answers[prior_message] = barney_message

data = pd.DataFrame({'questions': list(questions_and_answers.keys()), 'response': list(questions_and_answers.values())})
data.head()

tfidf = TfidfVectorizer(min_df=8, max_df = 0.05, ngram_range=(1, 1))
features = tfidf.fit_transform(data.questions + data.response)
question_vectors = tfidf.transform(data.questions)

user_input = input(">>> ")
while "exit" not in user_input.lower():
    user_input_tfidf = tfidf.transform([user_input])

    similarities = cosine_similarity(user_input_tfidf, question_vectors)
    idx = np.argsort(similarities)[0][-1]
    print(data.loc[idx, "response"])
    user_input = input(">>> ")


# print(f"Following episodes got ignored {ignored_episodes}")
# print(f"{len(ignored_episodes)} of 28 episodes got ignored. That is {round(len(ignored_episodes) / 208 * 100)}%")
# print(f"The corpus has {len(questions_and_answers)} pairs")
