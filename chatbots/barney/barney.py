import os
import config

questions_and_answers = {}
ignored_episodes = []

for file_name in os.listdir(config.processed_resources_folder_name):
    path = os.path.join(config.processed_resources_folder_name, file_name)
    
    if path.find("csv") == -1:
        continue

    
    with open(path, "r") as stream:
        lines = stream.read().splitlines()
        if len(lines) < 15:
            ignored_episodes.append(file_name)
            continue
    for line_index, line in enumerate(lines):
        index = line.find("Barney:")
        if index == 0:
            previous_line = lines[line_index-1]
            # wont work if previous line has no ":" or Barney has first line of an episode
            previous_line_text = previous_line[previous_line.find(":")+1:]
            questions_and_answers[previous_line_text] = line[len("Barney:"):]
    
for key, value in questions_and_answers.items():
    print(key, value, sep="               ", end="\n---------------------")