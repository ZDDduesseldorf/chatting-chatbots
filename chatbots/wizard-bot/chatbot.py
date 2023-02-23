from corpus_based.corpus_based import corpusbased_answer
from corpus_based.corpus_based import corpusbased_answer_daily_dialogue
from template_based.template_based import templatebased_answer
from template_based.template_based import special_patterns
from template_based.template_based import add_to_user_house_decision

import random

import phrases

import memory
from colorama import Fore, Style

# load language model
#nlp = spacy.load("en_core_web_lg")

repetition_memory = {'question 1' : 'answer1'}

count_inputs = 0

def get_answer_of_type(phrases):
    return random.choice(phrases)

def get_response(input: str):
    # every time the get_response() is called, add 1 to the counter
    # so we know how many inputs the user gave
    global count_inputs
    count_inputs += 1
    add_to_user_house_decision(input)

    if (count_inputs == 10):
        user_house = choose_user_house()
        print("Hm... right. I see ... you're' a " + user_house + "!")
        memory.add_user_name(user_house)
        memory.user_house = user_house
    
    answer = special_patterns(input)

    # Check if already asked
    if(answer == None):
        answer = get_modified_answer(input)

    # Template-Based-Pattern-Matching
    if(answer == None):
        answer = templatebased_answer(input)

    # Corpus-Based-Matching (Tfidf)
    if(answer == None):
        answer = corpusbased_answer(input, 0.58)

    # Corpus-Based-Matching 2 (Tfidf)
    if(answer == None):
        answer = corpusbased_answer_daily_dialogue(input, 0.58)

    # Standard/Fallback Answer
    if(answer == None):
        answer = get_answer_of_type(phrases.fallbacks)

    # add to memory (history)
    repetition_memory[input] = answer
    
    return answer

def get_modified_answer(input : str):
    # TODO check for similar input  instead of exact
    if (input in repetition_memory):
        if (input[-1] == '?'):
            return get_answer_of_type(phrases.repetitions_questions)
        else:
            return get_answer_of_type(phrases.repetitions_statements)
    return None

def choose_user_house():
    # get key with maximum value
    sum = 0

    # sum up all values to check percentages
    for key, value in memory.get_input_counts().items():
        print(key + ": " + str(value))
        sum += value
    
    # if sum == 0, fallback
    if (sum == 0):
        return "Gryffindor"

    if (memory.input_count["questions"] / sum > 0.75):
        return "Ravenclaw"
    
    if (memory.input_count["imperatives"] / sum > 0.75):
        return "Slytherin"

    if (memory.input_count["pleasantries"] / sum > 0.75):
        return "Hufflepuff"

    # if none of the above is matched -> Gryffindor
    return "Gryffindor"