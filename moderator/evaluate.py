from message import Message
import spacy
from typing import List

nlp = spacy.load("en_core_web_lg")


def check_sentence_simularity(full_conversation: List[Message], possible_next_messages: List[Message]):
    ###
    # lematizing? enable is_alpha? ueberpruefen wie man das vor der simularity berechnung enablen kann
    ###

    print('Quantity of possible answers:', len(possible_next_messages))
    ranked_messages: List[Message] = []
    if len(full_conversation) != 0:
        prev_sentence = full_conversation[-1].message
        doc1 = nlp(prev_sentence)  # spacy nlp

        for message in possible_next_messages:
            doc2 = nlp(message.message)
            simularity = doc1.similarity(doc2)
            print('Simularity: ', simularity)
            message.ranking_number += simularity

            # treshhold if sentences are to simular or loop happend
            if simularity > 0.95 or loop_checker(full_conversation, message.message):
                message.ranking_number = -5.0
            ranked_messages.append(message)
    return ranked_messages

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
