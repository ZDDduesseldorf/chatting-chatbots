import json
import random
import time
from threading import Timer

import pusher
import pysher
from bot import Bot
from message import Message

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
        self.channel = None
        self.elapsed = False
        self.answers = []
        self.bots = []
        self.init_connection()

    def make_elapsed(self):
        self.elapsed = True
        if len(self.answers) > 0:
            self.choose_answer()

    def choose_answer(self):
        selected = random.choice(self.answers)
        self.emit_message(selected)

    def wait_for_responses(self, data):
        data = json.loads(data)
        timeout = len(data["message"].split())
        self.channel.bind("chatbot_response", self.evaluate_response)
        Timer(timeout, self.make_elapsed).start()

    def emit_message(self, data):
        self.pusher_client.trigger(
            channels="chatting-chatbots", event_name="moderator_message", data=data
        )
        self.answers.clear()
        self.elapsed = False
        self.wait_for_responses(data)

    def evaluate_response(self, data):
        if self.elapsed is False:
            self.answers.append(data)
        else:
            self.emit_message(data)

    def init_chat(self):
        first_message = input("Message:")
        self.emit_message(
            Message(
                bot_id=0, bot_name="Starti", message=first_message
            ).to_json_event_string()
        )

    def register_chatbot(self, data):
        data = json.loads(data)
        bot = Bot(id=data["id"], name=data["name"], method=data["method"])
        self.bots.append(bot)
        self.pusher_client.trigger(
            channels="chatting-chatbots",
            event_name=f"moderator_connection_{bot.id}",
            data=bot.id,
        )
        print(
            f"{bot.name} ({bot.id}) joined the conversation. Active bots: {len(self.bots)}"
        )

    def connect_handler(self, _):
        self.channel = self.pysher_client.subscribe("chatting-chatbots")
        self.channel.bind("chatbot_connection", self.register_chatbot)
        Timer(10, self.init_chat).start()

    def init_connection(self):
        self.pysher_client.connection.bind(
            "pusher:connection_established", self.connect_handler
        )
        self.pysher_client.connect()


if __name__ == "__main__":
    moderator = Moderator()
    while True:
        time.sleep(1)
