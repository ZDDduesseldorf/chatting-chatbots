# Transformer Chatbot Sara Lacoste

_Authors: [Paul Kretschel](https://github.com/paulkre), [Kevin Zielke](https://github.com/Knerten0815)_

Chatbot system based on the transformer architecture published in [_Attention is all you need_](https://arxiv.org/abs/1706.03762) (2017).

**Sara Lacoste** was the name "chosen" by one of the better performing models we trained with this system:

![Birth of Sara Lacoste](https://github.com/ZDDduesseldorf/chatting-chatbots/blob/main/chatbots/sara_lacoste/docs/birth-of-sara-lacoste.png)

## Prepare Data

Create a CSV file containing your training data set and save it to `./data/<dataset-name>.csv`. The CSV file has to be in the following format and use semicolons (;) as seperators:

| Input | Output |
| - | - |
| What is the biggest fish? | A whale shark. |
| ... | ... |

There are some examples in the `./dataset_to_csv_scripts/` folder showing how to load and refactor data sets from [_🤗 Datasets_](https://huggingface.co/docs/datasets/index).
**Sara Lacoste** was trained on a data set that was merged from the following data sets:

- [_Yahoo Answers Dataset (from Kaggle)_](https://www.kaggle.com/datasets/jarupula/yahoo-answers-dataset)
- [_Chatbot Dataset Topical Chat (from Kaggle)_](https://www.kaggle.com/datasets/arnavsharmaas/chatbot-dataset-topical-chat)
- [_cornell_movie_dialog (from 🤗 Datasets)_](https://huggingface.co/datasets/cornell_movie_dialog)
- [_empathetic_dialogues (from 🤗 Datasets)_](https://huggingface.co/datasets/empathetic_dialogues)
- [_allenai/prosocial-dialog (from 🤗 Datasets)_](https://huggingface.co/datasets/allenai/prosocial-dialog)

The merged data set can be downloaded [_here_](https://fhd.sharepoint.com/:u:/r/teams/Chatbotsdiesmartsind/Freigegebene%20Dokumente/General/Datensaetze/merged.csv.zip?csf=1&web=1&e=jTSgxU).

## Training

Run the following command to start the training:

```
python train_model.py --dataset <dataset-name>
```

``<dataset-name>`` takes the name of the data sets CSV file, without the .csv ending.

## Evaluation

After you have trained your model, you can evaluate it by running the following command:

```
python evaluate_model.py --dataset <dataset-name>
```

This will load the model and ask it some standard questions. The questions and the models answers will be shown in the command prompt and saved to `./logs/<dataset-path>/test_chat_log.csv` for later comparison with other trainings. After that you can enter anything into the cammand line to chat and evaluate the model freely.

``<dataset-path>`` will be created during training and is made up by the models hyperparameters (see [Configuration](#configuration)) used for training.
``<dataset-path>`` follows this naming rule: \<*dataset-name*\>\_2<sup>\<*target-vocab-size-exp*\></sup>Voc\_\<*max-samples*\>Smp\_\<*max-length*\>Len\_\<*batch-size*\>Bat\_\<*buffer-size*\>Buf\_\<*num-layers*\>Lay\_\<*num-heads*\>Hed\_\<*epochs*\>Epo

[_Tensorboard_](https://www.tensorflow.org/tensorboard) is another way to evaluate the model. It can be used to monitor the loss and accuracy functions of the transformer model during training or compare them between different models after the training. Run:

```
tensorboard --logdir logs
```

Access http://localhost:6006/ through a browser for the tensorboard GUI.

## Configuration

You can change the default parameters used during training by passing the following command line arguments. Just remember to use the same values when evaluating your model afterwards.

| Argument | Description |
| - | - |
| `--dataset` | Name of the datasets CSV file, without the .csv ending |
| `--target-vocab-size-exp` | Exponent of the target size for the tokenizer vocabulary |
| `--max-samples` | Maximum number of data set samples used |
| `--max-length` | Maximum number of tokens allowed for each sample |
| `--buffer-size` | Buffer size for data set shuffling |
| `--batch-size` | Batch size of the data set during training |
| `--epochs` | Number of epochs during training |
| `--num-layers` | Number of layers in transformer model |
| `--num-units` | Number of units in transformer model |
| `--d-model` | Number of dimensions in transformer model |
| `--num-heads` | Number of heads in multi-head attention of transformer model |
| `--dropout` | Dropout rate during training |

## Let Sara chat with other bots

If you want to know what Sara and the other chatbots of this repository are gossiping about, you will need to set up [chatbotsclient](https://github.com/Robstei/chatbotsclient) and follow the installation instructions. After that you can run the following command:

```
python chat.py --dataset <dataset-name>
```
