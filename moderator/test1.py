import random
from typing import List

from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message


# TRANSFORMER EVALUATE
def respond(message: Message, conversations: List[Message]):
    return random.choice(
        [
            "I like butter.",
            "I love mochito.",
            "I prefer müsli.",
            "Avocados are my favorite.",
        ]
    )


if __name__ == "__main__":
    chatbot1 = Chatbot(respond, "Bot 1")
