## Transformer Chatbot

_Authors: Ben Kräling, Philipp Vogel_ <br>
Dataset preparation model and interface for a transformer based chatbot in TensorFlow and Keras.

### Setup

- create new anaconda environment and initialize environment `transformer-chatbot`

  ```
  conda create -n transformer-chatbot tensorflow-gpu python=3.9? //schauen ob python version ok und ob der Command klappt
  conda activate transformer-chatbot

  ```

- run installation script
  ```
  pip install -r requirements.txt
  ```

### Dataset

Merging following datasets resulted in more than 2 million question and answer samples:

- yahoo answers dataset (https://www.kaggle.com/datasets/jarupula/yahoo-answers-dataset)
- chatbot-dataset-topical-chat (https://www.kaggle.com/datasets/arnavsharmaas/chatbot-dataset-topical-chat)
- cornell_movie_dialog (https://huggingface.co/datasets/cornell_movie_dialog)
- empathetic_dialogues (https://huggingface.co/datasets/empathetic_dialogues)
- allenai/prosocial-dialog (https://huggingface.co/datasets/allenai/prosocial-dialog)

Merged dataset can be downloaded [here](https://fhd.sharepoint.com/:u:/t/Chatbotsdiesmartsind/EVipogxYkvxKpxbXpIOmXT4BqmakIh75tJmh2QACCOah4g?email=florian.huber%40hs-duesseldorf.de&e=eUGCMm) and should be saved in `data` //!!!! paths ändern

Dataset preparation:

- Split conversation pairs into a list of questions and answers.
- Pre-process each sentence by removing special characters.
- Trim sentence to prevent data loss for sentences longer than the configured max sentence length and filter out longer sentences which cannot be trimmed.
- Build and store tokenizer.
- Tokenize each sentence and add start_token and end_token.
- Pad tokenized sentences by adding zeros to the end of the sentence until max length is reached.

### Model

The model is based on the transformer technique described in the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

### Usage

#### 1. Create Dataset

```
python create_dataset.py
```

- the tokenized dataset will be saved to `data`. //!!! Hier noch Paths und Folder anpassen

#### 2. Train Model

```
python train_model.py
```

- the final (early stop) and best (lowest loss) trained model will be saved to `model`. //!!! Hier noch Paths und Folder anpassen

#### 3. Create test data as CSV

```
python testing.py
```

- the CSV file will be saved to `testing_results`.

#### 4. Chatbot interface

```
python chatbot.py
```

- to connect the chatbot with the moderator bot, you have to type in your pusher credentials ([see chatbotsclient README](https://github.com/Robstei/chatbotsclient))
