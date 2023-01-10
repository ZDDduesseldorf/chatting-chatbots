from typing import List
from chatbotsclient.message import Message
from chatbotsclient.chatbot import Chatbot
import helpers
import data
import transformer
from chatbot import TransformerChatbot
from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message
from typing import List


params = helpers.Params()
loader = data.DataLoader(params)
tokenizer = loader.get_tokenizer()

model = transformer.transformer(params, tokenizer.vocab_size)
model.load_weights(params.weights_path)

sara = TransformerChatbot(model, tokenizer, params.max_length)


def respond(message: Message, conversation: List[Message]):
    # custom answer computation of your chatbot
    answer = sara.predict(message.message)
    return answer


print("it's alive...")
chatbot = Chatbot(respond, "Sara Lacoste")
