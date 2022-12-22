import random
from typing import List

from chatbot import Chatbot
from message import Message


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
    chatbot = Chatbot(respond, "Philli")
