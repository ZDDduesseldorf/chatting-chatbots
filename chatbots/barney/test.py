import spacy
import pandas

my_dict = {"a":"A","b":"B"}
my_dict_with_index = None


df = pandas.DataFrame.from_dict(my_dict, orient="index")
print(df.head())

""" 

for index,(key, value) in enumerate(my_dict.items()):
    my_dict_with_index[index]
 """