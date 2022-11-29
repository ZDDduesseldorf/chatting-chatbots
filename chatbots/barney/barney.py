import os

questions_and_answers = {}

resources_directory_name = "resources"
processed_resources_folder_name = os.path.join(resources_directory_name, "processed")

for file_name in os.listdir(processed_resources_folder_name):
    path = os.path.join(processed_resources_folder_name, file_name)
    with open(path, "r") as stream:
        lines = stream.read().splitlines()

    for line_index, line in enumerate(lines):
        index = line.find("Barney:")
        if index == 0:
            previous_line = lines[line_index-1]
            # wont work if previous line has no ":" or Barney has first line of an episode
            previous_line_text = previous_line[previous_line.find(":")+1:]
            questions_and_answers[previous_line_text] = line[len("Barney:"):]
    
for key, value in questions_and_answers.items():
    print(key, value, sep="               ", end="\n---------------------")
