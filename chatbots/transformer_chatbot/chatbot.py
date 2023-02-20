from typing import List
import transformer as model
import sentence_processing
from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message


class TransformerChatbot:

    def __init__(self):
        self.conversation = []

    def postprocess(self, sentence):
        """Process sentence."""
        
        sentence = sentence_processing.remove_repetitions(sentence)
        sentence = sentence_processing.remove_spaces(sentence)
        sentence = sentence_processing.replace_after_sentence_sign(sentence)
        sentence = sentence_processing.handle_following_appostrophs(sentence)
        sentence = sentence[0].upper() + sentence[1:]
        return sentence

    def respond(self, sentence):
        """Respond to input."""

        output = self.postprocess(model.predict(sentence))
        return output


transformer = TransformerChatbot()

def respond(message: Message, conversation: List[Message]):
    """Creates answer for given input message."""
    answer = transformer.respond(message.message)
    return answer


chatbot = Chatbot(respond, "Theo", app_id="", app_key="",
                  app_secret="", app_cluster="eu")
