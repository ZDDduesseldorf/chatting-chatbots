import random
import re
from chatbots.chatbot_eliza_style.reflection import reflect
from chatbots.chatbot_eliza_style.text_patterns import psychobabble


def eliza_answer(user_input: str):
    """Generation of answers for ELIZA-like bot.

    Parameters
    ----------
    user_input
        String with user input for ELIZA to respond to.
    """

    # Test input string for all known text patter in pychobabble
    for pattern, responses in psychobabble:
        match = re.search(pattern.lower(), str(user_input).lower().strip())
        if match:
            answer = random.choice(responses)
            return answer.format(*[reflect(g.strip(",.?!")) for g in match.groups()])
    return None


def run_eliza_bot():
    """Starts ELIZA-like bot.
    """
    print("Hi! I am ELIZA. Kind of. What can I do for you?")
    user_input = ""
    while user_input.lower().strip() not in ["exit", "quit", "q"]:
        user_input = input(">> ")
        answer = eliza_answer(user_input)
        if answer is None:
            print("Mhhh. I am not sure if I can follow...")
        else:
            print(answer)

    print("Thanks for talking to me. Bye!")


if __name__ == '__main__':
    run_eliza_bot()
