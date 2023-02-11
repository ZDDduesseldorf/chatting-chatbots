# GPT Chatbot “Sara Lacoste”

_Authors: Paul Kretschel, Kevin Zielke_

Chatbot system based on the transformer architecture published in [_Attention is all you need_](https://arxiv.org/abs/1706.03762) (2017).

Why “Sara Lacoste”?

![Birth of Sara Lacoste](https://github.com/ZDDduesseldorf/chatting-chatbots/blob/optimus_fine/chatbots/transformer_chatbot/docs/birth-of-sara-lacoste.png)

## Training

Create a CSV file containing your training data set and save it to `./data/<dataset-name>.csv`. The file has to be in the following format:

| Input | Output |
| - | - |
| What is the biggest fish? | A whale shark. |
| ... | ... |

Run the following command to start the training:

```
$ python train_model.py --dataset <dataset-name>
```

