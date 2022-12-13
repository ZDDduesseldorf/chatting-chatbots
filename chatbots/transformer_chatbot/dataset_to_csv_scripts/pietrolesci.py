from datasets import load_dataset
import pandas as pd

split = "train"

""" dataset = load_dataset("pietrolesci/dialogue_nli", split=split)

dataset = dataset.rename_column("sentence1", "Input")
dataset = dataset.rename_column("sentence2", "Output")
dataset = dataset.remove_columns(["label", "triple1", "triple2", "dtype", "id", "original_label"])
nliDataset = dataset.to_pandas() """
nliDataset = pd.read_csv(rf"C:\Users\User\Desktop\transformer\chatting-chatbots\chatting-chatbots\chatbots\transformer_chatbot\data\pietrolesci_og_{split}.csv")
nliDataset.to_csv(rf"C:\Users\User\Desktop\transformer\chatting-chatbots\chatting-chatbots\chatbots\transformer_chatbot\data\pietrolesci_{split}.csv", index=False, sep=';')