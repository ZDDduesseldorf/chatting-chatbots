from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

split = "train"

dataset = load_dataset("empathetic_dialogues", split=split)

pdDataset = dataset.to_pandas()

data = {'Input': [], 
        'Output': [],
       }

df = pd.DataFrame(data)

skipIndex = 0

for i in tqdm(range(len(pdDataset)-1)):    
    if pdDataset.loc[i][0] == pdDataset.loc[i+1][0]:
        data = {'Input': [pdDataset.loc[i][5]], 
                'Output': [pdDataset.loc[i+1][5]],
               }
        df2 = pd.DataFrame(data, index=[i - skipIndex])
        df = pd.concat([df, df2])
    else:
        skipIndex += 1

df.to_csv(f"D:\Programmierung\StudyProjects\empathetic_dialogues\empathetic_dialogues_{split}.csv", index=False)