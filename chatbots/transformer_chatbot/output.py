import pyttsx3


class Output:
    """Text to speech instance"""

    def __init__(self, voice):
        self.conversation = []
        self.engine = pyttsx3.init()
        self.voices = []
        self.engine.setProperty("voice", voice)
        self.engine.setProperty("rate", 160)

    def speak(self, sentence):
        utterance = self.engine.say(sentence)
        utterance.runAndWait()
