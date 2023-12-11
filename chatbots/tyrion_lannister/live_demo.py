import os.path
import random

from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message
from typing import List

from model.patternbased_chatbot import PatternBasedChatbot
from model.spacy_chatbot import SpacyChatbot

names_memory = []


def compute_reply(message: str, source_name=None):
    if source_name is not None and source_name != '' and source_name not in names_memory:
        names_memory.append(source_name)
        corpus_based.add_known_person(source_name)

    # template based chatbot reply
    reply = specialized_patter_based(message)
    reply = randomly_append_chatbot_name(reply) if reply is not None else None

    if reply is not None:
        return reply

    # Corpus based chatbot reply
    reply, similarity = corpus_based(message)

    # template based chatbot reply
    if similarity < 0.90:
        reply = universal_patter_based(message)
        reply = randomly_append_chatbot_name(reply) if reply is not None else None

    return reply


def randomly_append_chatbot_name(reply, probability=42.0):
    if random.uniform(0.0, 100.0) <= probability:
        name = random.choice(names_memory)
        reply = reply[:-1] + ', {}'.format(name) + reply[-1]

    return reply



def respond(message: Message, conversation: List[Message]):
    return compute_reply(message.message, conversation)


if __name__ == "__main__":
    # define necessary paths
    specialized_text_pattern_file = os.path.normpath(os.getcwd() + '/Data/specialized_text_patterns.json')
    universal_text_pattern_file = os.path.normpath(os.getcwd() + '/Data/universal_text_patterns.json')

    # initialize necessary objects
    corpus_based = SpacyChatbot()
    specialized_patter_based = PatternBasedChatbot(specialized_text_pattern_file)
    universal_patter_based = PatternBasedChatbot(universal_text_pattern_file)

    print("Shall we endeavor to forge a link with other digital conversationalists?")
    print("Affirm with 'yes', and negate with 'no'... a simple enough task, I should think.")
    user_input = input(">>> ").lower().strip()

    if user_input == "yes":
        chatbot = Chatbot(respond, "TyrionBot")

    else:
        print("Very well, you have leave to speak, but choose your words wisely.")
        print('Well hello there, my dear friend. Might I trouble you for your name?')
        user_name = input(">>> ").strip()
        print('Delighted to meet you, {}. Is there something you need help with, or are you simply looking for a new friend to share a drink with?'.format(user_name))

        user_input = input(">>> ").strip()
        while "exit" not in user_input.lower():

            chatbot_reply = compute_reply(user_input, user_name)
            print(chatbot_reply)

            user_input = input(">>> ").strip()
