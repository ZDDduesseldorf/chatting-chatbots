#from datasets import load_dataset
import pandas as pd

filename1 = "pietrolesci_train"
filename2 = "pietrolesci_extra_train"
filename3 = "empathetic_dialogues_test"

filename_out = "piet_all"

data1 = pd.read_csv(rf"data/{filename1}.csv",  sep=';')
data2 = pd.read_csv(rf"data/{filename2}.csv", sep=';')
data3 = pd.read_csv(rf"data/{filename3}.csv", sep=';')

all_data = pd.concat([data1, data2, data3])

all_data.to_csv(rf"data/{filename_out}.csv", index=False, sep=';')
