from typing import List

from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message

import requests


def answer_function(message, conversation):
    public_url = "https://d8d0-2a01-4f8-1c1e-a475-00-1.eu.ngrok.io" #changes after manual vm restart
    rest_suburl = "/webhooks/rest/webhook"
    url = public_url+rest_suburl
    user_message = {'sender': 'test_user', 'message': message }
    response = requests.request("POST", url, json=user_message).json()
    return(response[0]['text'])

def respond(message: Message, conversation: List[Message]):
    # custom answer computation of your chatbot
    answer = answer_function(message.message, conversation)
    return answer


chatbot = Chatbot(respond, "Peter")
