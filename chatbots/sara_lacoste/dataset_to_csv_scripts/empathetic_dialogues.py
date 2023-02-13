from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

split = "test"

dataset = load_dataset("empathetic_dialogues", split=split)

pd_dataset = dataset.to_pandas()

data = {
    'Input': [],
    'Output': [],
}

df = pd.DataFrame(data)

skip_index = 0

for i in tqdm(range(len(pd_dataset)-1)):
    if pd_dataset.loc[i][0] == pd_dataset.loc[i+1][0]:
        data = {
            'Input': [pd_dataset.loc[i][5].replace("_comma_", ",")],
            'Output': [pd_dataset.loc[i+1][5].replace("_comma_", ",")],
        }
        df2 = pd.DataFrame(data, index=[i - skip_index])
        df = pd.concat([df, df2])
    else:
        skip_index += 1

df.to_csv(
    rf"data/empathetic_dialogues_{split}.csv",
    index=False,
    sep=';',
)
