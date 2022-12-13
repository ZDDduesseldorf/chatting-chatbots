#from datasets import load_dataset
import pandas as pd

filename1 = "empathetic_dialogues_train"
filename2 = "empathetic_dialogues_validation"
filename3 = "empathetic_dialogues_test"

filenameOut = "empathetic_dialogues_all"

data1 = pd.read_csv(rf"C:\Users\User\Desktop\transformer\chatting-chatbots\chatting-chatbots\chatbots\transformer_chatbot\data\{filename1}.csv",  sep=';')
data2 = pd.read_csv(rf"C:\Users\User\Desktop\transformer\chatting-chatbots\chatting-chatbots\chatbots\transformer_chatbot\data\{filename2}.csv", sep=';')
data3 = pd.read_csv(rf"C:\Users\User\Desktop\transformer\chatting-chatbots\chatting-chatbots\chatbots\transformer_chatbot\data\{filename3}.csv", sep=';')

allData = pd.concat([data1, data2, data3])

allData.to_csv(rf"C:\Users\User\Desktop\transformer\chatting-chatbots\chatting-chatbots\chatbots\transformer_chatbot\data\{filenameOut}.csv", index=False, sep=';')