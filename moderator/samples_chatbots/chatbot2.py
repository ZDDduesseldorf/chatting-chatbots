import random
from typing import List

from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message


def random_answer(message, conversation):
    return random.choice(
        [
            "I like bread.",
            "I love peanuts.",
            "I prefer porridge.",
            "Coconuts are my favorite.",
        ]
    )


def respond(message: Message, conversation: List[Message]):
    # custom answer computation of your chatbot
    answer = random_answer(message.message, conversation)
    return answer


chatbot = Chatbot(respond, "Pete", app_id="", app_key="", app_secret="", app_cluster="eu")
