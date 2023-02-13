# GPT Chatbot “Sara Lacoste”

_Authors: Paul Kretschel, Kevin Zielke_

Chatbot system based on the transformer architecture published in [_Attention is all you need_](https://arxiv.org/abs/1706.03762) (2017).

Why “Sara Lacoste”?

![Birth of Sara Lacoste](https://github.com/ZDDduesseldorf/chatting-chatbots/blob/optimus_fine/chatbots/transformer_chatbot/docs/birth-of-sara-lacoste.png)

## Training

Create a CSV file containing your training data set and save it to `./data/<dataset-name>.csv`. The CSV file needs to use semicolons (;) as seperators and has to be in the following format:

| Input | Output |
| - | - |
| What is the biggest fish? | A whale shark. |
| ... | ... |

There are some examples in the `./dataset_to_csv_scripts/` folder showing how to load and refactor datasets from [_🤗 Datasets_](https://huggingface.co/docs/datasets/index).

Run the following command to start the training:

```
python train_model.py --dataset <dataset-name>
```

## Evaluation

After you have trained your model, you can evaluate it by running the following command:

```
python evaluate_model.py --dataset <dataset-name>
```

## Configuration

You can change the default parameters used during training by passing the following command line arguments. Just remember to use the same values when evaluating you model afterwards.

| Argument | Description |
| - | - |
| `--dataset` | Name of the dataset |
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

