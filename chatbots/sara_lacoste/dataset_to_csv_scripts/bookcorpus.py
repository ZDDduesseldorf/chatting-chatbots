from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

split = "test"

dataset = load_dataset("bookcorpus")

pd_dataset = dataset['train'].to_pandas()

data = {
    'Input': [],
    'Output': [],
}

df = pd.DataFrame(data)


skip_index = 0

inputs = []
outputs = []
for i in tqdm(range(int(len(pd_dataset)/5))):  # -1)):
    inputs.append(pd_dataset.loc[i].text)
    outputs.append(pd_dataset.loc[i+1].text)

df = pd.DataFrame({'Input': inputs, 'Output': outputs})

df.to_csv(
    rf"data/books_fifth.csv",
    index=False,
    sep=';',
)
