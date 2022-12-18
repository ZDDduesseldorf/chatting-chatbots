import json


class Message():
    def __init__(self, message, bot_id, bot_name):
        self.__message = message
        self.__bot_id = bot_id
        self.__bot_name = bot_name
        self.__ranking_number = 0.0

    @property
    def message(self):
        return self.__message

    @property
    def bot_id(self):
        return self.__bot_id

    @property
    def bot_name(self):
        return self.__bot_name

    @property
    def ranking_number(self):
        return self.__ranking_number

    @ranking_number.setter
    def ranking_number(self, value):
        self.__ranking_number = value

    def toJSON_event_string(self) -> str:
        return json.dumps(
            {
                "type": "message",
                "message": self.message,
                "bot_id": self.bot_id,
                "bot_name": self.bot_name
            }
        )
