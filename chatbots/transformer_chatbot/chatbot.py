import re
from typing import List
import transformer as model
from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message


def handle_following_appostrophs(sentence):
    sentence = re.sub(r" s ", "'s ", sentence)
    sentence = re.sub(r" m ", "'m ", sentence)
    sentence = re.sub(r" re ", "'re ", sentence)
    sentence = re.sub(r" t ", "'t ", sentence)
    return sentence


def replace_after_sentence_sign(sentence):
    for index, char in enumerate(sentence):
        if char == "." or char == "?" or char == "!":
            # space counts as char
            if index + 2 < len(sentence):
                sentence = (
                    sentence[: index + 2]
                    + sentence[index + 2].upper()
                    + sentence[index + 3:]
                )
    return sentence


def remove_repetitions(sentence):
    words = sentence.split()
    for index, word in enumerate(words):
        if index < len(words) - 1 and word == words[index + 1]:
            print(index)
            sentence = sentence[:index] + sentence[-1]
            break

    return sentence


class TransformerChatbot:
    def __init__(self):
        self.conversation = []

    def process(self, sentence):
        """Process sentence."""
        sentence = remove_repetitions(sentence)
        sentence = (
            sentence.replace(" the u ", " the usa ")
            .replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
        )
        sentence = replace_after_sentence_sign(sentence)
        sentence = handle_following_appostrophs(sentence)
        sentence = sentence[0].upper() + sentence[1:]
        return sentence

    def respond(
        self,
        sentence,
    ):
        """Respond to input."""
        output = self.process(model.predict(sentence))
        return output


transformer = TransformerChatbot()


def respond(message: Message, conversation: List[Message]):
    answer = transformer.respond(message.message)
    return answer


chatbot = Chatbot(respond, "Fridolin")
