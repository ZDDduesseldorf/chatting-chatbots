## Transformer Chatbot

_Authors: Ben Kräling, Philipp Vogel_ <br>
Dataset preparation model and interface for a transformer based chatbot in TensorFlow and Keras.

### Setup

Create conda environment with necessary dependencies.
  ```
  conda create --name transformer-chatbot --file requirements.txt
  conda activate transformer-chatbot
  ```

### Dataset

Merging following datasets resulted in more than 2 million question and answer samples:

- yahoo answers dataset (https://www.kaggle.com/datasets/jarupula/yahoo-answers-dataset)
- chatbot-dataset-topical-chat (https://www.kaggle.com/datasets/arnavsharmaas/chatbot-dataset-topical-chat)
- cornell_movie_dialog (https://huggingface.co/datasets/cornell_movie_dialog)
- empathetic_dialogues (https://huggingface.co/datasets/empathetic_dialogues)
- allenai/prosocial-dialog (https://huggingface.co/datasets/allenai/prosocial-dialog)

Merged dataset can be downloaded [here](https://fhd.sharepoint.com/:u:/t/Chatbotsdiesmartsind/EVipogxYkvxKpxbXpIOmXT4BqmakIh75tJmh2QACCOah4g?email=florian.huber%40hs-duesseldorf.de&e=eUGCMm) and should be saved in `data`

Dataset preparation:

- Split conversation pairs into a list of questions and answers.
- Pre-process each sentence by removing special characters.
- Trim sentence to prevent data loss for sentences longer than the configured max sentence length and filter out longer sentences which cannot be trimmed.
- Build and store tokenizer.
- Tokenize each sentence and add start_token and end_token.
- Pad tokenized sentences by adding zeros to the end of the sentence until max length is reached.

### Model

The model is based on the transformer technique described in the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

<img src="https://user-images.githubusercontent.com/33390325/220045500-f79d01ed-b9df-4bde-a7d6-1763f5418dbb.jpeg" width="300">
<i>Source: Attention Is All You Need, Ashish Vaswani et al.</i><br/><br/>


Encoder and decoder (including single layers implementation) can be found in transformer.py. 
This contains following parts of the transformer archtieture:
- general transformer architecture
- embeddings of input tokens
- definition of encoder and decoder layer which are using:
  - multihead attention layers (masked and not masked)
  - feed forward neural networks
- stacking of encoder and decoder layers

All custom transformer layers used by encoder and decoder as well as masks are located in layers.py:
- positional encoding
- multihead attention including the scaled dot product
- look ahead mask for first attention layer of decoder
- padding mask for all layers

### Usage

All relevant parameters for data preperation and training are managed via env file.
|Parameter|Description|
|---------|-----------|
|MAX_SAMPLES|This parameter was used for testing purposes to receive fast results. Setting it to 0 results in max samples count available.|
|MAX_LENGTH|The max length of question and answer samples. Sentences longer than the provided value will be trimmed if possible.|
|MIN_LENGTH|This parameter was used in the experimental phase of the project. We wanted to evaluate sequential training with incresing sentence lengths.|
|BATCH_SIZE|The amount of samples passed to the neural network simultaniously.|
|BUFFER_SIZE||
|NUM_LAYERS|Amount of encoder and decoder layers.|
|D_MODEL|Model dimension.|
|NUM_HEADS|Amount of concatinated multi head attention layers.|
|UNITS|Amount of units in the feed forward neural networks.|
|DROPOUT|Dropout value for all model layers.|
|EPOCHS|Amount of epochs. Since early stopping is activated, the training will probibly consist of less epochs though.|

#### 1. Create Dataset

```
python create_dataset.py
```

- the tokenized dataset will be saved to `data`.

#### 2. Train Model

```
python train_model.py
```

- the final (early stop) and best (lowest loss) trained model will be saved to `model`.

#### 3. Create test data as CSV

```
python testing.py
```

- the CSV file will be saved to `testing_results`.

#### 4. Chatbot interface

```
python chatbot.py
```

- to connect the chatbot with the moderator bot you have to provide pusher credentials ([see chatbotsclient README](https://github.com/Robstei/chatbotsclient))
