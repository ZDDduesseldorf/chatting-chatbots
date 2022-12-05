import websockets
from websockets import server
from websockets.legacy.server import WebSocketServerProtocol
import asyncio
import json
import random
from typing import List

connections = []
conversation = []
shutdown_event = None

desired_conversation_len = 10
first_message = "test"
desired_chatbot_amount = 2

async def chose_next_message(possible_next_messages):
    #placeholder. currently a random answer is chosen as the next sentece
    next_answer = possible_next_messages[random.randint(0, len(possible_next_messages)-1)]["message"]
    return next_answer

async def handler(websocket: WebSocketServerProtocol):
    try:
        global conversation
        is_main_handler = False
        connection = {"websocket": websocket, "id": len(connections)}
        connections.append(connection)
        print(f"connection number: {connection['id']}")
        
        if len(connections) < desired_chatbot_amount:
            # websockets get closed when the handler ends.
            # Hence every handler except for the last one are awaiting the shutdown_event indefinetly
            print(f"connection {connection} is waiting for shutdown")
            await shutdown_event.wait()
        else:
            is_main_handler = True

        if is_main_handler:
            print(f"{connection['id']} is starting")
            
            # send each client it's id
            for connection in connections:
                await connection["websocket"].send(json.dumps({"type": "start", "id": connection["id"]}))

            while len(conversation) < desired_conversation_len:
                last_message = first_message if len(conversation) == 0 else conversation[-1]
                responses = []
                # This loop might be better as a TaskGroup
                for connection in connections:
                    await connection["websocket"].send(json.dumps({"type": "message", "message": last_message}))
                    response = json.loads(await connection["websocket"].recv())
                    responses.append(response)
                next_sentence = await chose_next_message(responses)
                conversation.append(next_sentence)
                print(f"conersation is {len(conversation)} long")
                print(f"last line in conversation: {conversation[-1]} from {conversation}")

            shutdown_event.set()
    finally:
        connections.remove(connection)
        if len(connections) == 0:
            # create file output of conversation
            #filename = f"conversation_
            #open()
            shutdown_event.clear()
            conversation = []

import datetime;
current_time = datetime.datetime.now()
print(current_time)

async def main():
    global shutdown_event
    shutdown_event = asyncio.Event()
    async with server.serve(handler,"", 8080):
        await asyncio.Future()

asyncio.run(main())

print("test")
