from moderator_bot import Message
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
        prev_sentence=full_conversation[-1].message 
        doc1 = nlp(prev_sentence) #spacy nlp
        
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