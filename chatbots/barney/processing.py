import csv
import os
import re

from config import (
    csv_quotechar,
    csv_separator,
    processed_resources_folder_name,
    scraped_resources_folder_name,
)

for file_name in sorted(os.listdir(scraped_resources_folder_name)):
    input_path = os.path.join(scraped_resources_folder_name, file_name)
    with open(input_path, "r", encoding="uft-8") as stream:
        lines = stream.read().splitlines()

    # remove lines with instructions
    lines = filter(lambda line: False if re.search(r"^\(.*\)$", line) else True, lines)

    # remove instructions in the middle of text and the space in front of it
    lines = list(map(lambda line: re.sub(r" \(.*\)", "", line), lines))

    lines = list(map(lambda line: re.sub(r"^\[.*\]$", csv_separator, line), lines))

    questions_and_answers = {}
    for line_index, line in enumerate(lines):
        index = line.find("Barney:")
        if index == 0:
            previous_line = lines[line_index - 1]

            # skip because Barney had the first line of the scene
            if previous_line == csv_separator:
                continue

            # remove the name of the person speaking prior to Barney (and follwing collon and space)
            # don't cut text if there is no collon
            text_start = (
                previous_line.find(":") + 2 if previous_line.find(":") > 0 else 0
            )
            previous_line_text = previous_line[text_start:]

            # remove Barney's name and the folowing collon and space
            questions_and_answers[previous_line_text] = line[len("Barney: ") :]

    output_path = os.path.join(processed_resources_folder_name, file_name + ".csv")
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(
            file,
            delimiter=csv_separator,
            quotechar=csv_quotechar,
            quoting=csv.QUOTE_ALL,
        )
        writer.writerow(["prior_message", "barney_message"])
        writer.writerows(questions_and_answers.items())
