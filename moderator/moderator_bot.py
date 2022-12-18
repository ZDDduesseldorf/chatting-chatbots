import asyncio
import csv
import datetime
import json
import os
import random
from typing import List

from evaluate import (check_conversation_shares, check_sentence_simularity,
                      select_highest_rated_message)
from message import Message
from mock_conversation import all_possible_message, full_conversation
from websockets import server
from websockets.legacy.server import WebSocketServerProtocol

connections = []
SHUTDOWN_EVENT = None

DESIRED_CONVERSATION_LEN = 10
FIRST_MESSAGE_TEXT = "test"
DESIRED_CHATBOT_AMOUNT = 2
CONVERSATION_LOGS_DIRECTORY = "conversation"

conversation: List[Message] = []


async def choose_next_message(full_conversation: List[Message], possible_next_messages: List[Message]):

    # add ranking points based on sentence simularities
    messages_ranked_by_simularity = check_sentence_simularity(
        full_conversation, possible_next_messages)
    # factor message rankings based on message frequency
    messages_ranked_by_conversation_shares = check_conversation_shares(full_conversation,
                                                                       messages_ranked_by_simularity)

    # chose message with the higest ranking
    next_message = select_highest_rated_message(
        messages_ranked_by_conversation_shares)
    return next_message


async def handler(websocket: WebSocketServerProtocol):
    try:
        global conversation
        is_main_handler = False
        connection = {"websocket": websocket, "id": len(connections)}
        connections.append(connection)
        print(f"connection number: {connection['id']}")

        if len(connections) < DESIRED_CHATBOT_AMOUNT:
            # websockets get closed when the handler ends.
            # Hence every handler except for the last one are awaiting the shutdown_event indefinetly
            print(f"connection {connection} is waiting for shutdown")
            await SHUTDOWN_EVENT.wait()
        else:
            is_main_handler = True

        if is_main_handler:
            print(f"{connection['id']} is starting")

            # send each client it's id
            for connection in connections:
                await connection["websocket"].send(json.dumps({"type": "start", "bot_id": connection["id"]}))

            while len(conversation) < DESIRED_CONVERSATION_LEN:
                last_message = Message(FIRST_MESSAGE_TEXT, "", "") if len(
                    conversation) == 0 else conversation[-1]
                responses: List[Message] = []
                # This loop might be better as a TaskGroup
                for connection in connections:
                    await connection["websocket"].send(last_message.to_json_event_string())
                    response_raw = json.loads(await connection["websocket"].recv())
                    response = Message(
                        response_raw["message"], response_raw["bot_id"], response_raw["bot_name"])
                    responses.append(response)
                next_message = await choose_next_message(conversation, responses)
                conversation.append(next_message)
                print(f"conversation is {len(conversation)} long")
                print(
                    f"last line in conversation: {conversation[-1]} from {conversation}")

            SHUTDOWN_EVENT.set()
    finally:
        connections.remove(connection)
        if len(connections) == 0:
            if not os.path.exists(CONVERSATION_LOGS_DIRECTORY):
                os.makedirs(CONVERSATION_LOGS_DIRECTORY)

            now = datetime.datetime.now()
            now_as_string = now.strftime("%Y-%m-%d--%H-%M-%S")

            filename = f"{now_as_string}.csv"
            path = os.path.join(CONVERSATION_LOGS_DIRECTORY, filename)

            with open(path, "a+", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                writer.writerow(["message", "bot_id", "bot_name"])
                for message in conversation:
                    writer.writerow(
                        [message.message, message.bot_id, message.bot_name])
            SHUTDOWN_EVENT.clear()
            conversation = []


async def main():
    global SHUTDOWN_EVENT
    SHUTDOWN_EVENT = asyncio.Event()
    async with server.serve(handler, "", 8080):
        await asyncio.Future()

asyncio.run(main())

# if __name__ == "__main__":
#     # Testing area
#     for message in check_sentence_simularity(full_conversation, all_possible_message):
#         print(f'Message: {message.message}, Ranking: {message.ranking_number}, BotID: {message.bot_id} ')
