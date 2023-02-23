from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from entity_recognition import replace_entities
import yaml
import pandas as pd 
import numpy as np
import pickle
from colorama import Fore, Style
import spacy


nlp = spacy.load("en_core_web_lg")

# Load the contents of the YAML file into a dictionary
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# read jsons
data = pd.read_json("corpus_based/data.json")
data.columns = ["questions", "responses"]

data_daily_dialogue = pd.read_json("corpus_based/data_daily_dialogue.json")
data_daily_dialogue.columns = ["questions", "responses"]

# load the vectors from the file
with open("corpus_based/vectors.pkl", "rb") as f:
    question_vectors = pickle.load(f)

with open("corpus_based/vectors_daily_dialogue.pkl", "rb") as f:
    question_vectors_daily_dialogue = pickle.load(f)

# Access the values in the configuration
min_df = config["TFIDF"]["min_df"]
max_df = config["TFIDF"]["max_df"]
ngram_range = (config["TFIDF"]["ngram_range_min"], config["TFIDF"]["ngram_range_max"])

# define tfidfs
tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range, stop_words=['english'], sublinear_tf=True)
tfidf.fit_transform(data.questions + data.responses)

tfidf_2 = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range, stop_words=['english'], sublinear_tf=True)
tfidf_2.fit_transform(data_daily_dialogue.questions + data_daily_dialogue.responses)

def corpusbased_answer(user_input : str, required_similarity : float):
    user_input_tfidf = tfidf.transform([user_input])

    similarities = cosine_similarity(user_input_tfidf, question_vectors)

    idx = np.argsort(similarities)[0][-1]

    question = data.loc[idx, "questions"] 
    response = data.loc[idx, "responses"] 
    similarity = similarities[0][idx]

    # print(Fore.RED + "DEBUG Harry: Question: " + question + "\n Answer: " + response + "\n Similarity: " + str(similarity) + Style.RESET_ALL) #Debugging
    
    response = replace_entities(user_input, response)


    if(similarity < required_similarity):
        return None

    return response

def corpusbased_answer_daily_dialogue(user_input : str, required_similarity : float):
    user_input_tfidf = tfidf_2.transform([user_input])

    similarities = cosine_similarity(user_input_tfidf, question_vectors_daily_dialogue)

    idx = np.argsort(similarities)[0][-1]

    question = data_daily_dialogue.loc[idx, "questions"] 
    response = data_daily_dialogue.loc[idx, "responses"] 
    similarity = similarities[0][idx]
    
    # print(Fore.RED + "DEBUG GPT: Question: " + question + "\n Answer: " + response + "\n Similarity: " + str(similarity) + Style.RESET_ALL) #Debugging

    response = replace_entities(user_input, response)

    if(similarity < required_similarity):
        return None

    return response

