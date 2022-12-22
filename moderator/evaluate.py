from typing import List

import spacy
from message import Message

nlp = spacy.load("en_core_web_lg")


def check_sentence_similarity(
    full_conversation: List[Message], possible_next_messages: List[Message]
) -> List[Message]:
    """Check Conversation Shares"""
    if len(full_conversation) == 0:
        return possible_next_messages

    possible_next_messages_ranked: List[Message] = []
    prev_message = full_conversation[-1].message_lemma
    prev_message_doc = nlp(prev_message)  # spacy nlp

    for message in possible_next_messages:
        message_doc = nlp(message.message_lemma)
        similarity = prev_message_doc.similarity(message_doc)
        message.ranking_number += similarity
        # treshhold if sentences are to similar or loop happend
        if similarity > 0.95 or loop_checker(full_conversation, message.message):
            message.ranking_number = -5.0 + similarity
        possible_next_messages_ranked.append(message)

    return possible_next_messages_ranked


def loop_checker(
    full_conversation: List[Message],
    possible_next_message: Message,
    window_size: int = 3,
):
    """Checks if message was already found in conversation (depends on window_size)"""
    for message in full_conversation[-window_size:]:
        if message.message == possible_next_message:
            return True

    return False


def check_conversation_shares(
    full_conversation: List[Message], possible_message: List[Message]
):
    """Check Conversation Shares"""
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
            share = (
                bot_message_count[message.bot_id] / conversation_message_count
            ) * 100
            normalized = 1 - (share / 100)
            message.ranking_number = message.ranking_number * normalized
        else:
            message.ranking_number = message.ranking_number * 1
        ranked_messages.append(message)

    return ranked_messages


def lemmatize_messages(possible_messages: List[Message]) -> None:
    """Lemmatize Messages"""
    for message in possible_messages:
        lemmatized_message = ""
        message_doc = nlp(message.message)
        for token in message_doc:
            lemmatized_message = lemmatized_message + token.lemma_ + " "
        message.message_lemma = lemmatized_message.strip().lower()


def get_subjects_and_objects(sentence):
    """Get Subject and Objects"""
    sent = nlp(sentence)
    ret = []
    for token in sent:
        if "nsubj" in token.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            ret.append(str(sent[start:end]).lower())

        if "dobj" in token.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            ret.append(str(sent[start:end]).lower())

    return ret


def check_object_subject_similarity(
    full_conversation: List[Message],
    possible_next_message: Message,
    window_size: int = 5,
):
    """Check Object Subject Similarity"""
    # get subjects and objects of message
    msg_parts = get_subjects_and_objects(possible_next_message.message)

    # get subjects and objects of conversation
    conv_parts = []
    for message in full_conversation[-window_size:]:
        conv_parts.extend(get_subjects_and_objects(message.message))

    # remove irrelevant subjects and objects like i or they
    irrelevant_parts = [
        "i",
        "you",
        "they",
        "it",
        "he",
        "she",
        "we",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "their",
        "its",
        "his",
        "her",
        "our",
        "mine",
        "yours",
        "theirs",
        "ours",
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "yourselves",
        "themselves",
    ]

    for part in irrelevant_parts:
        while part in conv_parts:
            conv_parts.remove(part)
        while part in msg_parts:
            msg_parts.remove(part)

    # print("conv_parts:")
    # print(conv_parts)
    # print("msg_parts:")
    # print(msg_parts)

    # score sinks further down the conversation. 0 if no match
    score = 1
    for conv_part in conv_parts:
        for msg_part in msg_parts:
            if msg_part == conv_part:
                print("match! with score ", score)
                print("msg_part = ", msg_part)
                print("conv_part = ", conv_part)
                return score
        score = score * 0.9

    print("total score = ", 0)
    return 0


def select_highest_rated_message(ranked_messages: List[Message]):
    """Select Highest Rated Message"""
    highest_rated_message = ranked_messages[0]
    for message in ranked_messages[1:]:
        if message.ranking_number > highest_rated_message.ranking_number:
            highest_rated_message = message
    return highest_rated_message
