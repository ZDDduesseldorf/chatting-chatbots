from typing import List

from chatbot import Chatbot
from message import Message


# TRANSFORMER EVALUATE
def respond(message: Message, conversations: List[Message]):
    return "I like butter."


if __name__ == "__main__":
    chatbot = Chatbot(respond, "Philli")
