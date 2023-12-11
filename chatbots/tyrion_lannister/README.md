# Tyrion Lannister Chatbot
This chatbot (sometimes) has the personality of Tyrion Lannister from Game of Thrones. He is a witty, intelligent and zynical dwarf and one of the main characters of the show. He likes to have a drink or two, but you might figure that out for yourself.

## Data
The Tyrion Lannister chatbot is based on the following corpora with request-reply pairs:
- ~ 1.300 [Tyrions dialogs from the tv show](https://genius.com/artists/Game-of-thrones)
- ~ 210.000 [cornell movie-dialogs corpus](https://convokit.cornell.edu/documentation/movie.html)
- ~ 86.000 [daily-dialogs](http://yanran.li/dailydialog.html)

Additionally ~1.000 patternbased answers cover some more specific user inputs such as greetings, exchanging names and saying goodbye.

## Environment
Choose one way to initialize your environment:

### 1. Conda Environment

```
conda env create -f environment.yml
```

```
conda activate Chatbot
```

### 2. pip Environment

```
pip install -r requirements.txt
```

## Preprocessing
You need to preprocess the data in order to use the Tyrion chatbot. Execute the following:

```
python preprocessing.py --use_preset
```

You can choose different configurations for the preprocessing. Here is a list of all possible configurations with a short explanation: 

| Argument | Explanation |
| - | - |
| `--data_path` | Path to data dir. |
| `--keep_prior_data` | Does not delete previously preprocessed data. |
| `--use_preset` | Preprocesses the Preset Corpora set by developer. |
| `--with_all` | Preprocess all types of corpora. |
| `--with_got` | Preprocess the got transcripts. |
| `--with_cornell` | Preprocess the cornell movie-dialogs corpus. |
| `--with_parliament` | Preprocess the Parliament Question Time Corpus. |
| `--with_daily` | Preprocess the Daily Dialogs Corpus. |


## Start the chatbot
You can start the Tyrion chatbot by executing the following command:

```
python live_demo.py
```

The chatbot will ask if you want to connect to establish a connection to the other chatbots or if you want to use the normal chatting mode. 
In order to connect to the other chabots, you'll need to set up a moderator chatbot. Therefor reference the [chatbotsclient repository](https://github.com/Robstei/chatbotsclient) and follow the instructions.
After entering 'yes' or 'no' you are **ready to go**!

## How it works

This Chatbot uses two kinds of Chatbots to generate a reply for a request. The first kind is a pattern based Chatbot which uses regex to find a match in a Pattern Database. The Database usually contains multiple replies for a pattern and chooses it randomly. The second kind is a corpus based Chatbot which compares a given request to a corpus. Then the corpus based chatbot uses the most similar request in its corpus to determine the reply. We also use substitution of named entities in our corpus based Chatbot.

We utilize two patten based Chatbots (one specialized and one universal) in the process of generating a reply. The specialized Chatbot uses pattern which are specific for manually selected Scenarios so that out Chatbot behaves in a certain way which we specified. The universal pattern based Chatbot uses patterns which are more general but it is only used if the similarity from the corpus based Chatbot is less than 90%.
The corpus based Chatbot uses the Data described in the Data section of this Readme and determines a reply and a similarity for a given request.

![Our_Complete_Chatbot](https://user-images.githubusercontent.com/47336789/221637807-6fef7334-1317-4b2c-864b-0917d5564b75.png)

A visualization of our two kinds of Chatbots is given in the following:

![Our Pattern Based Chatbot](https://user-images.githubusercontent.com/47336789/221637786-b5dd588e-b418-4086-9297-9e478012fd34.png)
---
![Our Corpus Based Chatbot](https://user-images.githubusercontent.com/47336789/221637735-04d42311-17f1-42e4-84db-c460a04b8f88.png)



