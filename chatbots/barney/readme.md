# Barney

## Authors

- Sophie Raps
- Robin Steil

## Overview

Barney is corpus based chatbot based on the character Barney Stinson from the series How I Met Your Mother and Joey Tribbiani from Friends that was developed as part of the course "Chatbot echo chamber - Building conversational chatbots with Python" at the University of Applied Sciences Düsseldorf. The bot is designed to interact with users and respond to requests by providing fun and quick-witted answers based on Barney's personality and style.

## Used technologies and libraries

- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/): A Library for parsing HTML and XML documents to extract information from them.
- [sklearn (TfidfVectorizer)](https://scikit-learn.org/stable/): A machine learning library that contains various models and tools. TfidfVectorizer is a tool for converting text documents to a matrix of tf-idf features.
- [spacy](https://spacy.io/): A library for natural language processing in Python.

## Corpora

The chatbot uses 3 corpora

1. Barney Stinson's Dialogues from the How I Met Your Mother series.
2. Dialogues generated with ChatGPT mimicing Barney's speech patterns and character traits.
3. Joey Tribbiani's Dialogues from Friends.

### Webscraping of How I Met Your Mother

Automated transcript scraping crawls the webpage "https://transcripts.foreverdreaming.org/" to collect episode links and extracts the text of each transcript from the HTML document and saves it in a text file. The code uses the BeautifulSoup library to parse the HTML document and allows the name of the text file to be adjusted to ensure the files are in a meaningful order.

### Processing Scraped Files to CSV

The `process_scraped_files()` function processes the scraped raw files, removes non-Barney statements and name prefixes, and maps the dialogs into question-and-answer pairs, where the question matches the previous non-Barney sentence and the answer corresponds to Barney's theorem. The result is a dictionary containing all of the questions and answers given by Barney in the scraped scripts.

The `save_corpus` function saves the dictionary as a single CSV file in the file path defined in the global variable CORPUS_PATH_BARNEY.

### Additional Corpus from Friends

Due to the limited number of question-answer pairs available, we decided to expand the corpus to improve the quality of the answers even further.
Friends and How I Met Your Mother are sitcoms with similar plot structures and characters, in particular the character Joey Tribbiani is often compared to Barney Stinson. Therefore, the Barney Stinson corpus was supplemented with another corpus with sentences by Joey.

We used the Cornell Movie Dialogue Corpus, because it is a large dataset of movie scripts that contains a collection of conversations extracted from raw movie scripts. We used the Friends utterances JSON-File and filtered for answers from Joey Tribbiani and the corresponding sentences. They were saved to a CSV file to use as second corpus.

### ChatGPT generated Answeres

In order to make the answer-question pairs from the Friends corpus more similar to Barney Stinson, some answers were manually converted by ChatGPT in Barney's style. 

### Greetings and Fallbacks

In order to avoid unpredictable answers to banal questions such as greetings, the Corpus-based chatbot has been expanded to include a rule-based component. Before the chatbot accesses the corpora, greetings and predefined questions are matched and answered. If no sufficiently good answer is found in the corpora fallback answers will be used. This is to ensure that the chatbot provides appropriate and consistent answers, even to simple and general questions.

The "greet" function returns a random greeting response if the user's input contains a greeting.
The "template" function returns a random answer to a question if it is in the "questions" list.

The chatbot will try to find a suitable answer from its corpora. If no matching answer is found, a fallback answer is chosen and returned. These fallback responses are predefined responses that the chatbot returns as a last resort when it cannot find a suitable answer.

## Answer generation

The respond can be generated with either `spacy` oder `tfidf`.

In `respond()`, the entry point for response generation, the user's input is first checked for salutations or specific phrases to generate a quick response. If no matching answer is found, `get_best_response_from_corpus()` is called to generate the best answer from the corpora. If no answer is found, a random answer is returned from the FAILS_RESPONSES list.

### SpaCy

When `spacy` is used, the user's input is first parsed with spacy to create a document representation of the input. The corpus variable is an instance of the Corpus class that contains a list of items. Each entry contains both the text and the spacy document vector.

Then the `get_best_response_from_corpus()` function searches each entry in the corpus list and calculates the similarity between the user's input and each entry's `spacy` document vector. If the similarity is higher than the current highest similarity, the current highest similarity is updated and the response is set to this entry's response.

### Tfidf method

When the `tfidf` method is used, the text of the user input is transformed by the vectorizer to obtain a TFIDF-vector, and the similarity between this vector and the TFIDF-vector of each corpus entry is calculated.

### Entity replacement

The replace_entity method in the Barney class takes a string (reply), a new entity (new_entity), and a Corpus instance. It looks for the first recognized entity with the label "PERSON" in the string and replaces it with the new entity. If no matching entity is found, it returns the original string. The method is used to match the entity in the bots' response string to the current user's name.

## How to use Barney

Starting the chatbot using spacy:

```
python barney.py 
```

If the argument "tfidf" is provided, the bot will use tfidf for similarity calculation instead of spacy (default).

If the argument "moderator" is provided, the bot will connect to the moderator bot that was developed as part of the course.

If the moderator tag is not included in the command-line arguments, the user is prompted for input, which the chatbot then processes to emit a response. This process repeats itself until the user enters "exit".

Example: To start the bot in tfidf mode connect to the moderator:

```
python barney.py tfidf moderator
```

## Code stucture

The Barney class includes the `responde` method which combines the different techniques used and generates a string response based on an input

The Corpus class is responsible for loading and preparing the conversation data from  CSV files. The CSV files contain messages and their replies.
