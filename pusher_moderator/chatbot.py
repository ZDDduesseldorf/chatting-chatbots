import time

import pusher
import pysher

APP_ID = "1527636"
KEY = "66736225056eacd969c1"
SECRET = "dbf65e68e6a3742dde34"
CLUSTER = "eu"


def channel_callback(data):
    print(data)
    answer = input("Message: ")
    pusher_client.trigger(
        channels="chatting-chatbots", event_name="response", data=answer
    )


def connect_handler(data):
    channel = pysher_client.subscribe("chatting-chatbots")

    channel.bind("message", channel_callback)


if __name__ == "__main__":

    pusher_client = pusher.Pusher(
        app_id=APP_ID, key=KEY, secret=SECRET, cluster=CLUSTER
    )

    pysher_client = pysher.Pusher(
        key=KEY,
        secret=SECRET,
        cluster=CLUSTER,
        user_data={"type": "chatbot", "name": "Harry"},
    )

    pysher_client.connection.bind("pusher:connection_established", connect_handler)
    pysher_client.connect()

    while True:
        time.sleep(1)