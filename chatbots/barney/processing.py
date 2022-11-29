import os
import re

resources_directory_name = "resources"
scraped_resources_folder_name = os.path.join(resources_directory_name, "scraped")
processed_resources_folder_name = os.path.join(resources_directory_name, "processed")

for file_name in os.listdir(scraped_resources_folder_name):
    input_path = os.path.join(scraped_resources_folder_name, file_name)
    with open(input_path, "r") as stream:
        lines = stream.readlines()

    # remove lines with instructions
    lines = filter(lambda line: False if re.search(r"^\(.*\)$", line) else True, lines)
    
    # remove instructions in the middle of text and the space in front of it
    lines = map(lambda line: re.sub(r"\(.*\)", "", line), lines)

    output_path = os.path.join(processed_resources_folder_name, file_name)
    
    with open(output_path, "w") as stream:
        for line in lines:
            stream.write(line)

