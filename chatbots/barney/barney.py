import os
import config
import csv
from config import csv_quotechar, csv_separator

questions_and_answers = {}
ignored_episodes = []

for file_name in os.listdir(config.processed_resources_folder_name):
    path = os.path.join(config.processed_resources_folder_name, file_name)
    
    if path.find("csv") == -1:
        continue

    # check if file is to small
    with open(path, "r") as file:
        lines = file.readlines()
        if len(lines) < 15:
            ignored_episodes.append(file_name)
            continue

    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=csv_separator, quotechar=csv_quotechar)
        
        for prior_message, barney_message in reader:
            # skip header
            if reader.line_num == 1:
                continue
            questions_and_answers[prior_message] = barney_message

print(f"Following episodes got ignored {ignored_episodes}")
print(f"{len(ignored_episodes)} of 28 episodes got ignored. That is {round(len(ignored_episodes) / 208 * 100)}%")
print(f"The corpus has {len(questions_and_answers)} pairs")

print("first 10 pairs are:") 
for index, (key, value) in enumerate(questions_and_answers.items()):
    if index == 5:
        break
    print(f"prior message: {key}" )
    print(f"barney's message: {value}")