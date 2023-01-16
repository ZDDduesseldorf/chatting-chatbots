import csv
import os
import re
from typing import Dict, List

import config
import numpy as np
import pandas as pd
import spacy
from chatbotsclient.chatbot import Chatbot, Message
from config import csv_quotechar, csv_separator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from templates import *

corpus: Dict[str, str] = {}


def load_corpus():
    """Get conversation peaces from csv"""
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
            reader = csv.reader(
                csvfile, delimiter=csv_separator, quotechar=csv_quotechar
            )

            for prior_message, barneys_message in reader:
                # skip header
                if reader.line_num == 1:
                    continue
                corpus[prior_message] = barneys_message


def get_best_reply_from_corpus(message: str) -> str:
    """Get most fitting reply out of scraped conversation pieces"""
    corpus_df = pd.DataFrame(
        {"prior_message": list(corpus.keys()), "barneys_message": list(corpus.values())}
    )

    # TODO: Tokenize und am besten spacy nutzen statt die Gewichtung vom Corpus (evt. bessere Ergebnisse, da der Corpus noch zu klein ist)
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
    prior_messages_tfidf = tfidf.fit_transform(corpus_df.prior_message)
    message_tfidf = tfidf.transform([message])

    similarities = cosine_similarity(message_tfidf, prior_messages_tfidf)
    idx = np.argsort(similarities)[0][-1]  # 2226.00 -> max value
    sim = np.sort(similarities)[0][-1]  # 0.61

    if template(message) is not None:
        return template(message)

    if sim > 0.5:
        return corpus_df.loc[idx]["barneys_message"]

    else:
        return str(np.random.choice(FAILS_RESPONSES))


def replace_entity(reply: str, new_entity: str) -> str:
    """Replace first recognized person in choosen reply with new_entity"""
    nlp = spacy.load("en_core_web_lg")

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [{"label": "PERSON", "pattern": "Lily"}]
    ruler.add_patterns(patterns)
    # nlp.add_pipe("merge_entities") Probably not needed
    doc = nlp(reply)

    ent_to_replace = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            ent_to_replace = ent
            break

    if ent_to_replace:
        return re.sub(ent_to_replace.text, new_entity, reply)

    return reply


def respond(message: Message) -> str:
    """Entry point to chatbot. Generate response based on received message"""
    if greet(message.message) is not None:

        print(f"greet: {message.message}")
        return greet(message.message)
    print(f"message: {message.message}")
    reply = get_best_reply_from_corpus(message.message)
    print(f"reply 1: {reply}")
    reply = replace_entity(str(reply), message.bot_name)
    print(f"replay 2: {reply}")
    return reply


def greet(sentence: str) -> str:
    """
    If user's input is a greeting, return a greeting response
    """

    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return np.random.choice(GREET_RESPONSES)


def template(sentence: str) -> str:
    """
    If user's input is a greeting, return a greeting response
    """

    for question in questions:
        if sentence in question["question"]:
            return np.random.choice(question["answers"])


if __name__ == "__main__":
    load_corpus()
    Chatbot(respond, "Barney")
    user_input = input()
    while user_input != "exit":
        user_input_as_message = Message(id, user_input, 1, "user")
        response = respond(user_input_as_message)
        print(response)
        user_input = input()
