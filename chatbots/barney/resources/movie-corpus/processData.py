import json
import jsonlines
import csv

from config import (
    CSV_QUOTECHAR,
    CSV_SEPERATOR,
)

qa_dict = {}
try:
    with jsonlines.open("/Users/sophieraps/Downloads/friends-corpus/utterances.jsonl") as f:
         data = [line for line in f]

         for d in data:
            if d['reply-to'] is not None and d['speaker'] == "Joey Tribbiani":
                answer = d['text']
                question_id = d['reply-to']
            
                for d2 in data:
                    if d2['id'] == question_id:
                        question = d2['text']
                if question is not "" and answer is not "":
                    print(f"Question: {question} - Answer: {answer}")
                    qa_dict[question] = answer
    
    with open('fragen_antworten_joey.csv', "w", newline="", encoding="utf-8") as file:
        print("writing")
        writer = csv.writer(
            file,
            delimiter=CSV_SEPERATOR,
            quotechar=CSV_QUOTECHAR,
            quoting=csv.QUOTE_ALL,
        )
        writer.writerows(qa_dict.items())

except json.JSONDecodeError as e:
    print("JSONDecodeError: ", e)
