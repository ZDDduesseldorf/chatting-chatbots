question_and_answers = {}

for line_index, line in enumerate(lines):
    index = line.find("Barney:")
    if index == 0:
        previous_line = lines[line_index-1]
        # wont work if previous line has no ":" or Barney has first line of an episode
        previous_line_text = previous_line[previous_line.find(":")+1:]
        question_and_answers[previous_line_text] = line[len("Barney:"):]
    
for key, value in question_and_answers.items():
    print(key, value, sep="               ", end="\n---------------------")
