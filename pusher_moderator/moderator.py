import time
from threading import Timer

import pusher
import pysher

APP_ID = "1527636"
KEY = "66736225056eacd969c1"
SECRET = "dbf65e68e6a3742dde34"
CLUSTER = "eu"


class Moderator:
    def __init__(self):
        self.pusher_client = pusher.Pusher(
            app_id=APP_ID, key=KEY, secret=SECRET, cluster=CLUSTER
        )
        self.pysher_client = pysher.Pusher(
            key=KEY, secret=SECRET, cluster=CLUSTER, user_data={"type": "moderator"}
        )
        self.elapsed = False
        self.answers = []
        self.init_connection()

    def make_elapsed(self):
        self.elapsed = True
        if len(self.answers) > 0:
            self.choose_answer()

    def choose_answer(self):
        selected = self.answers[-1]
        print(self.answers)
        self.emit_message(selected)

    def wait_for_responses(self, data):
        print(data)
        timeout = len(data.message.split())
        channel = self.pysher_client.subscribe("chatting-chatbots")
        channel.bind("response", self.evaluate_response)
        Timer(timeout, self.make_elapsed).start()

    def emit_message(self, data):
        self.pusher_client.trigger(
            channels="chatting-chatbots", event_name="message", data=data
        )
        self.answers.clear()
        self.elapsed = False
        self.wait_for_responses(data)

    def evaluate_response(self, data):
        if self.elapsed is False:
            self.answers.append(data)
        else:
            self.emit_message(data)

    def connect_handler(self, _):
        message = input("Message: ")
        self.emit_message({"name": "Starti", "message": message})

    def init_connection(self):
        self.pysher_client.connection.bind(
            "pusher:connection_established", self.connect_handler
        )
        self.pysher_client.connect()


if __name__ == "__main__":
    moderator = Moderator()
    while True:
        time.sleep(1)
