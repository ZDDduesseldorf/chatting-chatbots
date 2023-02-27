import csv
import os
import re
from typing import Dict

from config import (
    CORPUS_PATH_BARNEY,
    CSV_QUOTECHAR,
    CSV_SEPERATOR,
    scraped_resources_folder_name,
)


def process_scraped_files() -> Dict[str, str]:
    """process the scraped raw script files"""
    messages_and_responses = {}

    for file_name in sorted(os.listdir(scraped_resources_folder_name)):
        input_path = os.path.join(scraped_resources_folder_name, file_name)
        with open(input_path, "r", encoding="utf-8") as stream:
            lines = stream.read().splitlines()

        # remove lines with instructions
        lines = filter(
            lambda line: False if re.search(r"^\(.*\)$", line) else True, lines
        )

        # remove instructions in the middle of text and the space in front of it
        lines = list(map(lambda line: re.sub(r" \(.*?\)", "", line), lines))

        lines = list(map(lambda line: re.sub(r"^\[.*\]$", CSV_SEPERATOR, line), lines))

        for line_index, line in enumerate(lines):
            index = line.find("Barney:")
            if index == 0:
                previous_line = lines[line_index - 1]

                # skip because Barney had the first line of the scene
                if previous_line == CSV_SEPERATOR:
                    continue

                # remove name of the person speaking prior to Barney (and follwing collon and space)
                # don't cut text if there is no collon
                text_start = (
                    previous_line.find(":") + 2 if previous_line.find(":") > 0 else 0
                )
                previous_line_text = previous_line[text_start:]

                # remove Barney's name and the folowing collon and space
                messages_and_responses[previous_line_text] = line[len("Barney: ") :]

    return messages_and_responses


def save_corpus(processed_files: Dict[str, str]):
    """Save corpus as single csv"""
    output_path = os.path.join(CORPUS_PATH_BARNEY)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(
            file,
            delimiter=CSV_SEPERATOR,
            quotechar=CSV_QUOTECHAR,
            quoting=csv.QUOTE_ALL,
        )
        writer.writerows(processed_files.items())


if __name__ == "__main__":
    proccessed_text = process_scraped_files()
    save_corpus(proccessed_text)
