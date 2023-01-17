import re
import sys
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import spacy
from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message
from corpus import Corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.pipeline.entityruler import EntityRuler
from templates import *


class Barney:
    def __init__(self, mode="spacy") -> None:
        self.mode = mode
        self.corpus_instance = Corpus()
        self.corpus = self.corpus_instance.corpus
        self.nlp = self.corpus_instance.nlp
        self.vectorizer = self.corpus_instance.vectorizer

        patterns = [
            {"label": "PERSON", "pattern": "Lily"},
            {"label": "PERSON", "pattern": "Ranjit"},
        ]
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(patterns)
        # nlp.add_pipe("merge_entities") Probably not needed

    def get_best_response_from_corpus(self, message: str) -> Tuple[str, float]:
        """Get most fitting reply out of scraped conversation pieces"""

        # TODO: Tokenize und am besten spacy nutzen statt die Gewichtung vom Corpus (evt. bessere Ergebnisse, da der Corpus noch zu klein ist)
        """
        similarities = cosine_similarity(message_tfidf, prior_messages_tfidf)
        idx = np.argsort(similarities)[0][-1]
        sim = np.sort(similarities)[0][-1]
        """
        best_spacy_reply: Tuple[str, float] = "", 0
        message_doc = self.nlp(message)
        for entry in self.corpus:
            similarity = entry.spacy_doc.similarity(message_doc)
            if entry.spacy_doc.has_vector == False:
                print(entry)

            if similarity > best_spacy_reply[1]:
                best_spacy_reply = entry.barneys_message, entry.spacy_doc.similarity(
                    message_doc
                )
        return best_spacy_reply

    def replace_entity(self, reply: str, new_entity: str) -> str:
        """Replace first recognized person in choosen reply with new_entity"""
        doc = self.nlp(reply)

        ent_to_replace = None
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                ent_to_replace = ent
                break

        if ent_to_replace:
            return re.sub(ent_to_replace.text, new_entity, reply)

        return reply

    def greet(self, sentence: str) -> Union[str, None]:
        """
        If user's input is a greeting, return a greeting response
        """

        for word in sentence.split():
            if word.lower() in GREET_INPUTS:
                return np.random.choice(GREET_RESPONSES)

    def template(self, sentence: str) -> Union[str, None]:
        """
        If user's input is a greeting, return a greeting response
        """

        for question in questions:
            if sentence in question["question"]:
                return np.random.choice(question["answers"])

    def respond(self, message: Message) -> str:
        """Entry point to chatbot. Generate response based on received message"""

        template_response = self.template(message.message)
        if template_response is not None:
            return template_response

        greeting_response = self.greet(message.message)
        if greeting_response is not None:
            return greeting_response

        response, simularity = self.get_best_response_from_corpus(message.message)
        if simularity > 0.5:
            return self.replace_entity(str(response), message.bot_name)
        else:
            return str(np.random.choice(FAILS_RESPONSES))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "spacy"

    barney = Barney(mode)
    # Chatbot(respond, "Barney")
    user_input = input()
    while user_input != "exit":
        user_input_as_message = Message(id, user_input, 1, "user")
        response = barney.respond(user_input_as_message)
        print(response)
        user_input = input()
