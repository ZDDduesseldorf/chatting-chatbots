from bs4 import BeautifulSoup
import requests

#build all urls
# get all texts
# get individuell lines
# create dictionary for barney lines, with all the lines prior to them as a key 

base_url = "https://transcripts.foreverdreaming.org/"

url_steps = list(range(0, 25, 25))
ep_links = []

for url_step in url_steps:
    page = requests.get(base_url + "viewforum.php?f=177&start=" + str(url_step))
    soup = BeautifulSoup(page.content, "html.parser")
    tds = soup.findAll("td", class_ = "topic-titles row2")
    for i, td in enumerate(tds):
        if i != 0: #first td is not a link to an episode
            ep_links.append(td.find("h3").find("a")["href"][2:])          


page = requests.get(base_url + ep_links[0])
soup = BeautifulSoup(page.content, "html.parser")
result_set = soup.find("div", class_ = "postbody").find_all("p")
lines = []
for result in result_set:
    if result.string != None: #remove text of empty p tags
        lines.append(result.string)

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
