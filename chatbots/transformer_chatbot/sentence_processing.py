import re


def preprocess_sentence(sentence):
    """Replace special characters and correct wrong punctuations."""
    sentence = sentence.lower().strip()
    # set dot at the end of sentence if there is no ?.!
    if re.search('[.!?]$',sentence) is None:
        sentence = sentence + '.'
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", sentence)
    sentence = sentence.strip()
    
    return sentence

def trim_sentence(sentence, max_sentence_length):
    """Trim sentence to prevent data loss for sentences longer than max_sentence_length."""
    words = sentence.split()
    append = True
    if len(words) > max_sentence_length:
                words = words[:max_sentence_length-1]
                append = False
                if "?" in words:
                    index = words.index("?")
                    if index > 0:
                        words = words[:index+1]
                        sentence = " ".join(words)
                        append = True
                else:
                    if "!" in words:
                        index = words.index("!")
                        if index > 0:
                            words = words[:index+1]
                            sentence = " ".join(words)
                            append = True
                    else:
                        if "." in words:
                            index = words.index(".")
                            if index > 0 and words[index-1] != 'www':
                                words = words[:index+1]
                                sentence = " ".join(words)
                                append = True
    return sentence, append

def handle_following_appostrophs(sentence):
    """Set appostroph if appostroph is missing."""
    sentence = re.sub(r" s ", "'s ", sentence)
    sentence = re.sub(r" m ", "'m ", sentence)
    sentence = re.sub(r" re ", "'re ", sentence)
    sentence = re.sub(r" t ", "'t ", sentence)
    return sentence


def replace_after_sentence_sign(sentence):
    """Set character uppercase after punctuation."""
    for index, char in enumerate(sentence):
        if char == "." or char == "?" or char == "!":
            # space counts as char
            if index + 2 < len(sentence):
                sentence = (
                    sentence[: index + 2]
                    + sentence[index + 2].upper()
                    + sentence[index + 3:]
                )
    return sentence


def remove_repetitions(sentence):
    """Remove words if there are looping."""
    words = sentence.split()
    for index, word in enumerate(words):
        if index < len(words) - 1 and word == words[index + 1]:
            sentence = sentence[:index] + sentence[-1]
            break

    return sentence


def remove_spaces(sentence):
    """Remove spaces before punctuations."""
    sentence = (
            sentence.replace(" the u ", " the usa ")
            .replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
        )
    return sentence