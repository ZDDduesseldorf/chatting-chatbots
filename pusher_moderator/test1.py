import random
from typing import List

from chatbot import Chatbot
from message import Message


# TRANSFORMER EVALUATE
def respond(message: Message, conversations: List[Message]):
    return random.choice(
        [
            "I like peanut.",
            "I love lemons.",
            "I prefer bread.",
            "Strawberries are my favorite.",
        ]
    )


if __name__ == "__main__":
    chatbot = Chatbot(respond, "Benni")
