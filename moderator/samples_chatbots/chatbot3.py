import random
from typing import List

from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message


def random_answer(message, conversation):
    return random.choice(
        [
            "Hi, my name is steve.",
        ]
    )


def respond(message: Message, conversation: List[Message]):
    # custom answer computation of your chatbot
    answer = random_answer(message.message, conversation)
    return answer


chatbot = Chatbot(respond, "Steve", app_id="", app_key="", app_secret="", app_cluster="eu")
