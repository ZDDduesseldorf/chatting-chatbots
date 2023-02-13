
from datasets import load_dataset
import pandas as pd

split = "train"

dataset = load_dataset("allenai/prosocial-dialog", split=split)

dataset = dataset.rename_column("context", "Input")
dataset = dataset.rename_column("response", "Output")
dataset = dataset.remove_columns(["rots", "safety_label", "safety_annotations", "safety_annotation_reasons", "source", "etc", "dialogue_id", "response_id", "episode_done"])
proSocial = dataset.to_pandas()
proSocial.to_csv(rf"C:\Users\User\Desktop\transformer\chatting-chatbots\chatting-chatbots\chatbots\transformer_chatbot\data\prosocial_{split}.csv", index=False, sep=';')