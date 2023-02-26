import json
import jsonlines

qa_dict = {}
try:
    with jsonlines.open("/Users/sophieraps/Downloads/friends-corpus/utterances.jsonl") as f:
         data = [line for line in f]

         for d in data:
            if d['reply-to'] is not None and d['speaker'] == "Joey Tribbiani":
                # question_ids[d['id']] = True
                # print(question_ids)
                answer = d['text']
                
                question_id = d['reply-to']
            
                for d2 in data:
                    if d2['id'] == question_id:
                        question = d2['text']
                if question is not "" and answer is not "":
                    print(f"Question: {question} - Answer: {answer}")
                    qa_dict[question] = answer
    
    with open('fragen_antworten_joey.json', 'w') as f:
        json.dump(qa_dict, f)

except json.JSONDecodeError as e:
    print("JSONDecodeError: ", e)


#     import json
# import jsonlines

# qa_dict = {}
# try:
#     with jsonlines.open("/Users/sophieraps/Downloads/friends-corpus/utterances.jsonl") as f:
#          data = [line for line in f][:10]
#          questions = {d['id']: d['text'] for d in data if d['reply-to'] is None}

#          for d in data:
#             if d['reply-to'] is not None:
#                 answer = d['text']
#                 question_id = d['reply-to']
                
#                 if question_id in questions:
#                     question = questions[question_id]
#                     qa_dict[question] = answer
#                     print(f"Question: {question} - Answer: {answer}")

#     with open('fragen_antworten.json', 'w') as f:
#         json.dump(qa_dict, f)

# except json.JSONDecodeError as e:
#     print("JSONDecodeError: ", e)




# if __name__ == "__main__":
#     one()


# welche ids sind fragen (!reply-to = null)
# finde die id
# finde die passende antwort
# speichere in neue json