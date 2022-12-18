from typing import List

import spacy
from message import Message

nlp = spacy.load("en_core_web_lg")


def check_sentence_simularity(full_conversation: List[Message], possible_next_messages: List[Message]) -> List[Message]:
    ###
    # lematizing? enable is_alpha? ueberpruefen wie man das vor der simularity berechnung enablen kann
    ###

    print('Quantity of possible answers:', len(possible_next_messages))
    if len(full_conversation) == 0:
        return possible_next_messages

    possible_next_messages_ranked: List[Message] = []
    prev_message = full_conversation[-1].message
    prev_message_doc = nlp(prev_message)  # spacy nlp

    for message in possible_next_messages:
        message_doc = nlp(message.message)
        simularity = prev_message_doc.similarity(message_doc)
        print('Simularity: ', simularity)
        message.ranking_number += simularity
        # treshhold if sentences are to simular or loop happend
        if simularity > 0.95 or loop_checker(full_conversation, message.message):
            message.ranking_number = -5.0
        possible_next_messages_ranked.append(message)

    return possible_next_messages_ranked

# checks if message was already found in conversation (depends on window_size)
def loop_checker(full_conversation: List[Message], possible_next_message: Message, window_size: int = 3):

    for message in full_conversation[-window_size:]:
        if message.message == possible_next_message:
            return True

    return False


def check_conversation_shares(full_conversation: List[Message], possible_message: List[Message]):
    ranked_messages: List[Message] = []

    conversation_message_count = len(full_conversation)
    bot_message_count = {}

    for message in full_conversation:
        if message.bot_id in bot_message_count:
            bot_message_count[message.bot_id] += 1
        else:
            bot_message_count[message.bot_id] = 1

    # final message ranking will be factored based on frequency of messages sent by bot_id
    for message in possible_message:
        if message.bot_id in bot_message_count:
            share = (bot_message_count[message.bot_id] /
                     conversation_message_count) * 100
            normalized = 1 - (share / 100)
            message.ranking_number = message.ranking_number * normalized
        else:
            message.ranking_number = message.ranking_number * 1
        ranked_messages.append(message)

    return ranked_messages


def select_highest_rated_message(ranked_messages: List[Message]):
    highest_rated_message = ranked_messages[0]
    for message in ranked_messages[1:]:
        if message.ranking_number > highest_rated_message.ranking_number:
            highest_rated_message = message
    return highest_rated_message
