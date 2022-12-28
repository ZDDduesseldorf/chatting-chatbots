import random
from typing import List

from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message


# TRANSFORMER EVALUATE
def respond(message: Message, conversations: List[Message]):
    return random.choice(
        [
            "I like bread.",
            "I love peanuts.",
            "I prefer porridge.",
            "Coconuts are my favorite.",
        ]
    )


if __name__ == "__main__":
    chatbot1 = Chatbot(respond, "Bot 2")
