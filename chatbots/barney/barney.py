import re
import sys
from typing import Callable, List, Literal, Tuple, Union

import numpy as np
from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message
from config import CORPUS_PATH_BARNEY, CORPUS_PATH_BARNEY_CHATGPT, CORPUS_PATH_JOE
from corpus import Corpus
from sklearn.metrics.pairwise import cosine_similarity
from spacy.pipeline.entityruler import EntityRuler, PatternType
from templates import *

COMPARE_MODES = Union[Literal["spacy"], Literal["tfidf"]]


class Barney:
    def __init__(
        self, compare_mode: COMPARE_MODES = "spacy", log: bool = False
    ) -> None:
        self.compare_mode = compare_mode
        self.corpora: List[Corpus] = []
        self.add_corpora(
            [
                (CORPUS_PATH_BARNEY, "Barney"),
                (CORPUS_PATH_BARNEY_CHATGPT, "Barney_ChatGPT"),
                (CORPUS_PATH_JOE, "Joe"),
            ]
        )
        self.log = log

        if self.log:
            print(f"startet Barney in {compare_mode} mode ")

    def add_corpora(self, corpus_arguments_list: List[Tuple[str, str]]):
        for corpus_arguments in corpus_arguments_list:
            corpus = Corpus(corpus_arguments[0], corpus_arguments[1])
            # Type Error without explicit typing, when calling add_patterns
            patterns: List[PatternType] = [
                {"label": "PERSON", "pattern": "Lily"},
                {"label": "PERSON", "pattern": "Ranjit"},
            ]
            ruler = corpus.nlp.add_pipe("entity_ruler", before="ner")
            if isinstance(ruler, EntityRuler):
                ruler.add_patterns(patterns)
            self.corpora.append(corpus)

    def get_best_response_from_corpus(
        self, message: str, corpus: Corpus
    ) -> Tuple[str, float]:
        """Get most fitting reply out of scraped conversation pieces"""

        if self.compare_mode == "spacy":
            message_doc = corpus.nlp(message)
            best_spacy_reply: Tuple[str, float] = "", 0
            for entry in corpus.corpus:
                if not entry.spacy_doc.vector_norm:
                    print(entry)

                similarity = entry.spacy_doc.similarity(message_doc)
                if similarity > best_spacy_reply[1]:
                    best_spacy_reply = (
                        entry.barneys_message,
                        similarity,
                    )
            return best_spacy_reply

        elif self.compare_mode == "tfidf":
            message_tfidf = corpus.vectorizer.transform([message])
            best_tfidf_reply: Tuple[str, float] = "", 0
            for entry in corpus.corpus:
                similarity = cosine_similarity(entry.tfidf_vector, message_tfidf)[0][0]
                if similarity > best_tfidf_reply[1]:
                    best_tfidf_reply = (
                        entry.barneys_message,
                        similarity,
                    )
            return best_tfidf_reply

        return ("", 0)

    def replace_entity(self, reply: str, new_entity: str, corpus: Corpus) -> str:
        """Replace first recognized person in choosen reply with new_entity"""
        doc = corpus.nlp(reply)

        ent_to_replace = None
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                ent_to_replace = ent
                break

        if ent_to_replace:
            if self.log:
                print(f"replaced {ent_to_replace.text} with {new_entity}")
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

    def respond(self, message: Message, conversation: List[Message]) -> str:
        """Entry point to responding. Generate response based on received message"""

        template_response = self.template(message.message)
        if template_response is not None:
            if self.log:
                print("answer from templates")
            return template_response

        greeting_response = self.greet(message.message)
        if greeting_response is not None:
            if self.log:
                print("answer from greetings")
            return greeting_response

        for corpus in self.corpora:
            response, simularity = self.get_best_response_from_corpus(
                message.message, corpus
            )
            if simularity > 0.4:
                if self.log:
                    print(
                        f"answer from corpus: {corpus.name}. simularity: {simularity}"
                    )
                return self.replace_entity(response, message.bot_name, corpus)

        # return back answer
        if self.log:
            print("backup answer")
        return str(np.random.choice(FAILS_RESPONSES))


def respond_without_self_generator(
    mode: COMPARE_MODES,
) -> Callable[[Message, List[Message]], str]:
    barney = Barney(mode, log=True)

    def respond_without_self(message: Message, conversation: List[Message]) -> str:
        return barney.respond(message, conversation)

    return respond_without_self


if __name__ == "__main__":
    mode = "tfidf" if "tfidf" in sys.argv else "spacy"

    if "moderator" in sys.argv:
        Chatbot(
            respond_without_self_generator(mode),
            "Barney",
            app_id="1527636",
            app_key="66736225056eacd969c1",
            app_secret="dbf65e68e6a3742dde34",
            app_cluster="eu",
        )

    else:
        barney = Barney(mode, log=True)
        user_input = input()
        while user_input != "exit":
            user_input_as_message = Message(id, user_input, 1, "user")
            response = barney.respond(user_input_as_message, [])
            print(response)
            user_input = input()
