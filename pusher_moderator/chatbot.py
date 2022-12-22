import time
from typing import Callable, List, Union

import pusher
import pysher
from message import Message

APP_ID = "1527636"
KEY = "66736225056eacd969c1"
SECRET = "dbf65e68e6a3742dde34"
CLUSTER = "eu"


class Chatbot:
    def __init__(
        self,
        respond_method: Union[
            Callable[[Message, List[Message]], str], Callable[[Message], str]
        ],
        name: str,
    ):
        self.pusher_client = pusher.Pusher(
            app_id=APP_ID, key=KEY, secret=SECRET, cluster=CLUSTER
        )
        self.name = name
        self.pysher_client = pysher.Pusher(
            key=KEY,
            secret=SECRET,
            cluster=CLUSTER,
            user_data={"type": "chatbot", "name": self.name},
        )
        self.elapsed = False
        self.answers = []
        self.init_connection()
        self.respond_method = respond_method

        while True:
            time.sleep(1)

    def message_received(self, data):
        print(f"{data.name}: {data.message}")
        response = self.respond_method()
        self.pusher_client.trigger(
            channels="chatting-chatbots",
            event_name="response",
            data={"name": self.name, "message": response},
        )

    def connect_handler(self, _):
        channel = self.pysher_client.subscribe("chatting-chatbots")

        channel.bind("message", self.message_received)

    def init_connection(self):
        self.pysher_client.connection.bind(
            "pusher:connection_established", self.connect_handler
        )
        self.pysher_client.connect()


# TRANSFORMER EVALUATE
def respond(message: Message, conversations: List[Message]):
    print(message, conversations)
    return "I like peanut."


if __name__ == "__main__":
    chatbot = Chatbot(respond, "Benni")
