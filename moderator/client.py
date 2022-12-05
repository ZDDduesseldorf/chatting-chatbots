from websockets import client
from websockets.legacy.server import WebSocketServerProtocol
import asyncio
import json
import functools

async def handler(websocket, response_function, call_with_conversation):
    response_counter = 0
    id = None
    conversation = []
    async for message in websocket:
        print(message)
        event = json.loads(message)
        if event["type"] == "start":
            id = event["id"]
        if event["type"] == "message":
            message = event['message'] 

            if call_with_conversation:
                response_message = response_function(message, conversation)
                conversation.append[message]
            else:
                response_message = response_function(message)

            response = {
                "message": response_message,
                "response_counter": response_counter,
                "id" : id
            }

            print(f"sending message {response}")
            response_counter += 1
            await websocket.send(json.dumps(response))

async def main(response_function, call_with_conversation):
    bound_handler = functools.partial(handler, response_function=response_function,call_with_conversation=call_with_conversation)
    async with client.connect("ws://localhost:8080") as websocket:
        await bound_handler(websocket)

# every chatbot needs to implement this function and call it 
def chat(response_function, call_with_conversation=False):
    asyncio.run(main(response_function, call_with_conversation))
    
if __name__ == "__main__":
    chat(lambda x:x)
