import os
import re
import csv


OUTPUT_PATH = "data/transformer_data.csv"

def extract_line_data():
    with open("data/movie_dialog_corpus/movie_lines.tsv", encoding="utf-8") as file:
        for line in file:
            line = re.sub(r'\n$', "", line)

            if line[0] == '"':
                line = re.sub(r'""', '"', line[1:-1])

            fields = line.split("\t")

            output = fields[4] if len(fields) <= 5 else " ".join(fields[4:])

            yield fields[0], output


def extract_conversations():
    with open("data/movie_dialog_corpus/movie_conversations.tsv", encoding="utf-8") as file:
        for line in file:
            yield re.search(r'\[([^]]*)\]', line).group(1)[1:-1].split("' '")


print("> Loading CSV data ...")

utterances = []
line_id_to_utterance = {}

for line_id, utterance in extract_line_data():
    utterances.append(utterance)
    line_id_to_utterance[line_id] = utterance


def extract_samples():
    inputs = []
    outputs = []

    max_samples = int(os.getenv("MAX_SAMPLES", "0"))
    for line_ids in extract_conversations():
        for i in range(len(line_ids) - 1):
            inputs.append(line_id_to_utterance[line_ids[i]])
            outputs.append(line_id_to_utterance[line_ids[i + 1]])
            if max_samples and len(inputs) >= max_samples:
                return inputs, outputs
    return inputs, outputs


inputs, outputs = extract_samples()

print('> Writing data to "{}" ...'.format(OUTPUT_PATH))

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    writer = csv.writer(file, lineterminator='\n', delimiter=';')
    writer.writerow(["Input", "Output"])
    writer.writerows(zip(inputs, outputs))
