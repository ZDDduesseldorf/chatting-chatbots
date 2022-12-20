import time

import pusher
import pysher

APP_ID = "1527636"
KEY = "66736225056eacd969c1"
SECRET = "dbf65e68e6a3742dde34"
CLUSTER = "eu"

answers = []


def evaluate(message):
    # conversation shares, simalarity, ... calculation
    ranked_message = message
    return ranked_message


def choose_answer():
    return answers[0]


def evaluate_response(message):
    ranked_message = evaluate(message)
    answers.append(ranked_message)
    pusher_client.trigger(
        channels="chatting-chatbots", event_name="message", data=choose_answer()
    )


def connect_handler(data):
    channel = pysher_client.subscribe("chatting-chatbots")
    # timeout 3s
    channel.bind("response", evaluate_response)


if __name__ == "__main__":
    pusher_client = pusher.Pusher(
        app_id=APP_ID, key=KEY, secret=SECRET, cluster=CLUSTER
    )

    pysher_client = pysher.Pusher(
        key=KEY, secret=SECRET, cluster=CLUSTER, user_data={"type": "moderator"}
    )

    pysher_client.connection.bind("pusher:connection_established", connect_handler)
    pysher_client.connect()

    initial_input = input("Message: ")
    pusher_client.trigger(
        channels="chatting-chatbots", event_name="message", data=initial_input
    )

    while True:
        time.sleep(1)
