from typing import List
from chatbotsclient.message import Message
from chatbotsclient.chatbot import Chatbot
from lib.helpers import Params
from lib.data import DataLoader
from lib.transformer import create_model
from lib.chatbot import TransformerChatbot


params = Params()
loader = DataLoader(params)
tokenizer = loader.get_tokenizer()

model = create_model(params, tokenizer.vocab_size)
model.load_weights(params.weights_path)

sara = TransformerChatbot(model, tokenizer, params.max_length)


def respond(message: Message, conversation: List[Message]):
    # custom answer computation of your chatbot
    answer = sara.predict(message.message)
    return answer


print("it's alive...")
chatbot = Chatbot(respond, "Sara Lacoste")
