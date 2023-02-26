# Barney

## Authors

- Sophie Raps
- Robin Steil

## Overview

A corpus based chatbot based on the character Barney Stinson from the series "How I Met Your Mother" whichs that was developed as part of the course "Chatbot echo chamber - Building conversational chatbots with Python" at the University of Applied Sciences Düsseldorf. The bot is designed to interact with users and respond to requests by providing fun and quick-witted answers based on Barney's personality and style.

## Used technologies and libraries

- BeautifulSoup: a Library for parsing HTML and XML documents to extract information from them.
- sklearn (TfidfVectorizer): a machine learning library that contains various models and tools. TfidfVectorizer is a tool for converting text documents to a matrix of TF-IDF features.
- spacy: a library for natural language processing in Python.

## Source of corpora

The chatbot was trained using a corpus of text and dialogue from the How I Met Your Mother series to mimic Barney's speech patterns and character traits. The bot uses a combination of rules and machine learning to respond to user input and generate an appropriate response.

### Webscraping of How I Met Your Mother

Automated transcript scraping crawls the webpage "https://transcripts.foreverdreaming.org/" to collect episode links and extracts the text of each transcript from the HTML document and saves it in a text file. The code uses the BeautifulSoup library to parse the HTML document and allows the name of the text file to be adjusted to ensure the files are in a meaningful order.

### Processing Scraped Files to CSV

The `process_scraped_files()` function processes the scraped raw files, removes non-Barney statements and name prefixes, and maps the dialogs into question-and-answer pairs, where the question matches the previous non-Barney sentence and the answer corresponds to Barney's theorem. The result is a dictionary containing all of the questions and answers given by Barney in the scraped scripts.

The save_corpus function saves the dictionary as a single CSV file in the file path defined in the global variable CORPUS_PATH_BARNEY.

### Additional Corpus from Friends

Due to the fact that only 40% of the episodes could be scraped successfully and we only had a limited number of question-answer pairs available, we decided to expand the corpus to improve the quality of the answers.
Friends and How I Met Your Mother are sitcoms with similar plot structures and characters, in particular the character Joey Tribbiani is often compared to Barney Stinson. Therefore, the Barney Stinson corpus was supplemented with another corpus with sentences by Joey.

We used the Cornell Movie Dialogue Corpus, because it is a large dataset of movie scripts that contains a collection of conversations extracted from raw movie scripts. We used the Friends utterances JSON-File and filtered for answers from Joey Tribbiani and the corresponding sentences. They were saved to a CSV file to use as second corpus.

### ChatGPT generated Answeres

In order to make the answer-question pairs from the Friends corpus more similar to Barney Stinson, some answers were manually converted by ChatGPT in Barney's style. However, this process was very time consuming and could not be automated, which is why only 150 sentences were so converted and saved as a corpus.

### Greetings and Fallbacks

In order to avoid unpredictable answers to banal questions such as greetings, the Corpus-based chatbot has been expanded to include a rule-based component. Before the chatbot accesses the corpora, greetings, fallbacks, and predefined questions are matched and answered. This is to ensure that the chatbot provides appropriate and consistent answers, even to simple and general questions.

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

When the `tfidf` method is used, the text of the user input is transformed by the vectorizer to obtain a TFIDF vector, and the similarity between this vector and the TF-IDF vector of each corpus entry is calculated.

### Entity replacement

The replace_entity method in the Barney class takes a string (reply), a new entity (new_entity), and a Corpus instance. It looks for the first recognized entity with the label "PERSON" in the string and replaces it with the new entity. If no matching entity is found, it returns the original string. The method is used to match the entity in the bots' response string to the current user's name.

## How to use Barney

The chatbot can run either as a participant of the moderator bot or as a standalone interactive chatbot.

If the "moderator" tag is included in the command line arguments, it can communicate with the moderator bot. In this case, credentials are also provided to assist the moderator.

If the "moderator" tag is not included in the command-line arguments, the user is prompted for input, which the chatbot then processes and emits a response. This process repeats itself until the user enters "exit".

To determine which method to use to compare the similarity between the user's input and the stored conversation chunks, the tag `spacy` or `tfidf`can be entered on the command line at startup. If no tag is used, `spacy` is used automatically.

## Code stucture

     -    Barney Class
    - Corpus class
    - scarping scripts

The Barney class includes a method called `get_best_response_from_corpus()` that responds with either the `spacy` or `tfidf` method. Both methods are ways of computing similarity to find the best answer for a given input.

The Corpus class is responsible for loading and preparing the conversation data from a CSV file. The CSV file contains the previous message and Barney's reply. The class uses spaCy to tokenize and parse the messages, and `sklearn` to create a TfIdf vector for each message in the corpus. The class stores the previous messages, Barney's reply, and the associated tf-idf vectors in a list of CorpusEntry objects.
