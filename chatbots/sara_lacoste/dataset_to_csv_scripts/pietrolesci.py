import pandas as pd

split = "train"

"""
dataset = load_dataset("pietrolesci/dialogue_nli", split=split)

dataset = dataset.rename_column("sentence1", "Input")
dataset = dataset.rename_column("sentence2", "Output")
dataset = dataset.remove_columns(["label", "triple1", "triple2", "dtype", "id", "original_label"])
nli_dataset = dataset.to_pandas()
"""
nli_dataset = pd.read_csv(rf"data/pietrolesci_og_{split}.csv")
nli_dataset.to_csv(rf"data/pietrolesci_{split}.csv", index=False, sep=';')
