import os
import re
from config import processed_resources_folder_name, scraped_resources_folder_name, csv_separator

for file_name in sorted(os.listdir(scraped_resources_folder_name)):
    input_path = os.path.join(scraped_resources_folder_name, file_name)
    with open(input_path, "r") as stream:
        lines = stream.read().splitlines()

    # remove lines with instructions
    lines = filter(lambda line: False if re.search(r"^\(.*\)$", line) else True, lines)
    
    # remove instructions in the middle of text and the space behind it
    lines = list(map(lambda line: re.sub(r"\(.*\) ", "", line), lines))

    lines = list(map(lambda line: re.sub(r"^\[.*\]$", csv_separator, line), lines))

    questions_and_answers = {}
    for line_index, line in enumerate(lines):
        index = line.find("Barney:")
        if index == 0:
            previous_line = lines[line_index-1]
            
            # skip because Barney had the first line of the scene
            if previous_line == csv_separator:
                continue

            # remove the name of the person speaking prior to Barney (and follwing collon and space)
            previous_line_text = previous_line[previous_line.find(":")+2:]

            # remove Barney's name and the folowing collon and space
            questions_and_answers[previous_line_text] = line[len("Barney: "):]

    output_path = os.path.join(processed_resources_folder_name, file_name)
    with open(output_path, "w") as stream:
        for key, value in questions_and_answers.items():
                stream.write(key + csv_separator + value + "\n")
# for key, value in questions_and_answers.items():
#     print(key, value, sep="               ", end="\n---------------------")