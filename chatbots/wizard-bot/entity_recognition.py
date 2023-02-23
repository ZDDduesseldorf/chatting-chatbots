import spacy
from spacy.pipeline import EntityRuler
import custom_entities
import re
import memory
import random


nlp = spacy.load("en_core_web_lg")

ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(custom_entities.entity_patterns)

def replace_entities(input : str, output : str):
    output_nlp = nlp(output)
    input_nlp = nlp(input)

    # Pre-Define entities
    entities = {
        "PERSON": memory.user_names,
        "ORG": custom_entities.org_patterns,
        "LOC": custom_entities.loc_patterns,
        "NORP": custom_entities.norp_patterns,
        "PRODUCT": custom_entities.product_patterns,
        "MONEY": custom_entities.money_patterns
    }

    # update pre-defined entities by entities in input
    for input_ent in input_nlp.ents:
        entities[input_ent.label_] = [input_ent.text]

    replaced_ents = {} # list that includes all already replaced ents with their replacement
    # Entity Recognition for entities in input
    for output_ent in output_nlp.ents:
        try:
            if output_ent.text in replaced_ents:
                # If this entity text has already been replaced, use the existing replacement string
                replacement = replaced_ents[output_ent.text]
            else:
                # If this is a new entity text, generate a new replacement string and add it to the dictionary
                replacement_ent = random.choice(entities[output_ent.label_]) #get random of entities

                if(len(entities[output_ent.label_])>1):
                    # remove from entities only if list won't be empty
                    entities[output_ent.label_].remove(replacement_ent)

                if (isinstance(replacement_ent, str)):
                    replacement = replacement_ent
                else:
                    # only if replacement_ent is not of type str
                    replacement = replacement_ent["pattern"]
                

                replaced_ents[output_ent.text] = replacement # save the replacement in replaced_ents
            
            # Replace the entity text with the replacement string
            output = re.sub(output_ent.text, replacement, output)
        except KeyError:
            pass

    return output

def get_entities(input : str):
    input_nlp = nlp(input)

    person = None
    org = None
    
    # returns last entity that matched this label
    for ent in input_nlp.ents:
        if ent.label_ == "PERSON":
            person = ent.text
        elif ent.label_ == "ORG":
            org = ent.text
    return person, org

def replace_entity(output : str, ent, new_ent, label: str):
    if ent.label_ == label:
        output = re.sub(ent.text, new_ent, output)
    return output

def replace_entity_custom(output, list, entity):
    for element in list:
        if element in output:
            output = re.sub(element, entity, output)
    return output