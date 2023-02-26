# Barney

## Authors

- Sophie Raps
- Robin Steil

## Overview

A corpus based chatbot based on the character Barney Stinson from the series "How I Met Your Mother" whichs that was developed as part of the course "Chatbot echo chamber - Building conversational chatbots with Python" at the University of Applied Sciences Düsseldorf. The bot is designed to interact with users and respond to requests by providing fun and quick-witted answers based on Barney's personality and style.

## Used technologies and libraries

- BeautifulSoup
-

## Source of corpora

The chatbot was trained using a corpus of text and dialogue from the How I Met Your Mother series to mimic Barney's speech patterns and character traits. The bot uses a combination of rules and machine learning to respond to user input and generate an appropriate response.

### Webscraping of How I Met Your Mother

Automated transcript scraping crawls the webpage "https://transcripts.foreverdreaming.org/" to collect episode links and extracts the text of each transcript from the HTML document and saves it in a text file. The code uses the BeautifulSoup library to parse the HTML document and allows the name of the text file to be adjusted to ensure the files are in a meaningful order.

### Processing Scraped Files to CSV

The process_scraped_files() function processes the scraped raw files, removes non-Barney statements and name prefixes, and maps the dialogs into question-and-answer pairs, where the question matches the previous non-Barney sentence and the answer corresponds to Barney's theorem. The result is a dictionary containing all of the questions and answers given by Barney in the scraped scripts.

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

- spacy
- tfidf
- entity replacement

## How to use Barney

## Code stucture

     -    Barney Class
    - Corpus class
    - scarping scripts
