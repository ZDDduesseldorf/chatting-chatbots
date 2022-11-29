from config import processed_resources_folder_name, csv_separator
#pip install -U spacy
#python -m spacy download en_core_web_md

import spacy
nlp = spacy.load("en_core_web_md") # English-language model

text = "The bats saw the cats with best stripes hanging upside down by their feet."
doc = nlp(text)

tokens_lemma_spacy = []
for token in doc:
    tokens_lemma_spacy.append(token.lemma_)

# print(tokens_lemma_spacy)

#pip install pandas
import pandas as pd
import os

#TODO: change the data
data = pd.DataFrame(columns=["questions", "answers"])
for file_name in os.listdir(processed_resources_folder_name):
    input_path = os.path.join(processed_resources_folder_name, file_name)
    # skip file if it is empty
    if os.stat(input_path).st_size == 0:
        continue

    file_data = pd.read_csv(input_path, sep=csv_separator)
    file_data.columns = ["questions", "answers"]
    data += file_data
    print(data)
# data.head()
# data.describe()

#pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1, 3))
# tfidf = TfidfVectorizer(min_df=4, max_df = 0.2, ngram_range=(1, 3))
features = tfidf.fit_transform(data.questions + data.answers)
features.shape #?

question_vectors = tfidf.transform(data.questions)
question_vectors[2] #?

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

### Test chatbot with static data
#user_input = "Hi what are you doing?"
# user_input_tfidf = tfidf.transform([user_input])
# similarities = cosine_similarity(user_input_tfidf, question_vectors)
# idx = np.argsort(similarities)[0][::-1]
# user_input_tfidf.shape, question_vectors.shape
# print(data.loc[idx, "questions"])

def response(input):
    user_input_tfidf = tfidf.transform([input])
    similarities = cosine_similarity(user_input_tfidf, question_vectors)
    idx = np.argsort(similarities)[0][-1]
    return data.loc[idx, "answers"]


user_input = input(">>> ")
while "exit" not in user_input.lower():
    print(response(user_input))
    user_input = input(">>> ")