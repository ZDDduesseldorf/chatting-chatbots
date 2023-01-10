import random
import re
from sys import platform
from typing import List

import pyttsx3
import transformer as model
from AppKit import NSSpeechSynthesizer
from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message


def isMac():
    if platform == "darwin":
        return True
    return False


def handle_following_appostrophs(sentence):
    sentence = re.sub(r" s ", "'s ", sentence)
    sentence = re.sub(r" m ", "'m ", sentence)
    sentence = re.sub(r" re ", "'re ", sentence)
    sentence = re.sub(r" t ", "'t ", sentence)
    return sentence


def replace_after_sentence_sign(sentence):
    for index, char in enumerate(sentence):
        if char == "." or char == "?" or char == "!":
            # space counts as char
            if index + 2 < len(sentence):
                sentence = (
                    sentence[: index + 2]
                    + sentence[index + 2].upper()
                    + sentence[index + 3 :]
                )
    return sentence


def remove_repetitions(sentence):
    words = sentence.split()
    for index, word in enumerate(words):
        if index < len(words) - 1 and word == words[index + 1]:
            print(index)
            sentence = sentence[:index] + sentence[-1]
            break

    return sentence


class TransformerChatbot:
    def __init__(self):
        self.conversation = []
        self.engine = pyttsx3.init()
        self.voices = []
        if isMac():
            for voice in NSSpeechSynthesizer.availableVoices():
                if "en-GB" in voice or "en-US" in voice:
                    self.voices.append(voice)
            self.voices.append("com.apple.speech.synthesis.voice.Whisper")
            selected_voice = random.choice(self.voices)
            self.engine.setProperty("voice", selected_voice)
        self.engine.setProperty("rate", 160)

    def speak(self, sentence):
        self.engine.say(sentence)
        self.engine.runAndWait()

    def process(self, sentence):
        """Process sentence."""
        sentence = remove_repetitions(sentence)
        sentence = (
            sentence.replace(" the u ", " the usa ")
            .replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
        )
        sentence = replace_after_sentence_sign(sentence)
        sentence = handle_following_appostrophs(sentence)
        sentence = sentence[0].upper() + sentence[1:]
        return sentence

    def respond(
        self,
        sentence,
    ):
        """Respond to input."""
        output = self.process(model.predict(sentence))
        print("Theo:", output)
        self.conversation.append(input)
        self.conversation.append(output)
        self.speak(output)
        return output


transformer = TransformerChatbot()

while True:
    inp = input("Message: ")
    transformer.respond(inp)


def respond(message: Message, conversation: List[Message]):
    answer = transformer.respond(message.message)
    return answer


chatbot = Chatbot(respond, "Jens")
