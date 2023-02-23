from chatbotsclient.chatbot import Chatbot
from chatbotsclient.message import Message
from typing import List
from chatbot import get_response
from chatbot import get_answer_of_type
from colorama import Fore, Style

import phrases

def compute(input: str, conversation: List[Message]):
    return get_response(input)
    
def respond(message: Message, conversation: List[Message]):
    answer = compute(message.message, conversation)
    return answer

# MAIN METHOD
if __name__ == '__main__':
    print(Fore.RED + "Do you want to establish a connection to other chatbots?" + Style.RESET_ALL + " Answer with 'yes' for yes and 'no' for no...")
    user_input = input(">>> ").strip()
        
    
    if(user_input == "yes"):
        chatbot = Chatbot(respond, "SnapeBot")

    else:
        log = open('log.txt', 'a')
        print(Fore.GREEN + get_answer_of_type(phrases.greetings) + Style.RESET_ALL)
        user_input = input(">>> ").strip()

        log.write('\nUser: ' + user_input)    

        while user_input.lower() not in phrases.exit_strings:
            
            answer = get_response(user_input)
            log.write('\nWizardBot: ' + answer)    

            print(answer + Style.RESET_ALL)
            print("________________________________________________")
            user_input = input(">>> ").strip()
            log.write('\nUser: ' + user_input)    

        exit_answer = get_answer_of_type(phrases.farewells)
        print(exit_answer)
        log.write('\nWizardBot: ' + exit_answer)
        log.write('\n_______________________________________')    