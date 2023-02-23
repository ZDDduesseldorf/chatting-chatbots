# WizardBot

This project is a chatbot built using Python and inspired by the Harry Potter universe. The chatbot is designed to have natural language conversations with users and respond to a variety of user inputs while trying to inpersonate Harry Potter as a character.

## Features

The chatbot currently has the following features:

- Greeting users with a random greeting
- Answering basic questions about itself
- Asking for more info about a named topic
- Providing a farewell message upon exiting the chat

## Requirements

To run the chatbot, you will need to have Python 3.x installed on your machine. You will also need to install the following Python packages:

- `nltk`
- `pandas`
- `spacy`
- `sklearn`
- `pickle`
- `colorama`
- `https://github.com/Robstei/chatbotsclient/releases/download/1.0.5/chatbotsclient-1.0.5.tar.gz`


You can install these packages using the following command:

```bash
pip install -r requirements.txt
```

Additionally, you need to install Spacy's language model `en_core_web_lg`

```bash
python -m spacy download en_core_web_lg
```

## Usage

To run the chatbot, navigate to the project directory and run the following command in your terminal:

```
python chatbot_client.py
```

The chatbot will then greet you and start a conversation. Firstly you have to enter if you want to establish a connection with other chatbots (yes/no). If so, WizardBot will connect with other chatbots through [chatbotsclient](https://github.com/Robstei/chatbotsclient). If not, you can start chatting with WizardBot yourself.

After that you can enter any natural language input and the chatbot will respond appropriately.

## Functionality

The chatbot utilizes a comprehensive workflow to provide accurate responses to the user's inputs. The chatbot_client.py file contains the main method, which initiates a loop until the user enters an exit message. The supported exit messages include:
    
    "exit", "quit", "q", "bye", "goodbye", "good bye", "adios", "ciao", "farewell", "finish", "done", "end", "stop", "halt", "cancel", "abort", and "close".

Upon receiving user input, the chatbot invokes the natural language processing module, chatbot.py, which uses the `get_response()` function. 
1. Firstly, the chatbot attempts to generate a template-based answer using the template_based.py's `templatebased_answer()` function. 
2. If no pattern is matched, the chatbot performs a corpus-based analysis of the Harry Potter movie transcripts using the corpus_based.py's `corpusbased_answer()` function, employing a tfidf vectorizer.

3. If the corpus-based analysis fails to provide an adequate response, the chatbot searches for a match in the DailyDialogue corpus using the corpus_based.py's `corpusbased_answer_2()` function, using a similar tfidf vectorizer approach.

4. Finally, if none of the above methods generate a suitable response, the chatbot defaults to providing a fallback response using the phrases.py module.

### Assigning a Hogwarts house
Additionally, the chatbot analyzes every user input through the template_based.py module's `add_to_user_house_decision()` function. After ten inputs from the user, the `choose_user_house()` function is executed, assigning a Hogwarts house to the user based on the gathered data. This information is utilized to refer to the user by their respective Hogwarts house.

All responses provided by the chatbot are centered around themes from the Harry Potter universe.

## Future Improvements

There are several areas for improvement in the chatbot, such as:

- Adding more corpus data (currently the first three movie scripts are added)
- Adding more conversational topics
- Implementing machine learning techniques to improve the chatbot's responses over time
- Integrating the chatbot with external APIs to provide more accurate and useful information

## Contribution

If you would like to contribute to the project, feel free to fork the repository and submit a pull request with your changes. Any contributions are welcome and appreciated!

## Acknowledgments

- Thanks to the [NLTK](https://www.nltk.org/) and [Spacy](https://spacy.io/) Python packages for providing useful tools for natural language processing and API requests, respectively.
- Thanks to the open-source community for providing inspiration and resources for building chatbots.
