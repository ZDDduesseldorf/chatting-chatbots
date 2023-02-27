import os

import requests
from bs4 import BeautifulSoup
from config import processed_resources_folder_name, scraped_resources_folder_name

for path in [scraped_resources_folder_name, processed_resources_folder_name]:
    if not os.path.isdir(path):
        os.makedirs(path)

BASE_URL = "https://transcripts.foreverdreaming.org/"

url_steps = list(range(0, 201, 25))
episode_links = {}

# Collect episode links in dictionary.
# Keys are formatted season number  and episode number(example: 01x01)
for url_step in url_steps:
    page = requests.get(
        BASE_URL + "viewforum.php?f=177&start=" + str(url_step), timeout=1
    )
    soup = BeautifulSoup(page.content, "html.parser")
    tds = soup.findAll("td", class_="topic-titles row2")
    for i, td in enumerate(tds):
        if i != 0:  # first td is not a link to an episode
            link = td.find("h3").find("a")
            link_episode_name = link.string[0 : link.string.index(" ")]
            link_url = BASE_URL + link["href"][2:]
            episode_links[link_episode_name] = link_url

# create a txt file per collected link
for episode_name, episode_link in episode_links.items():
    page = requests.get(episode_link, timeout=1)
    soup = BeautifulSoup(page.content, "html.parser")
    result_set = soup.find("div", class_="postbody").find_all("p")
    filepath = os.path.join(scraped_resources_folder_name, episode_name)
    with open(filepath, "w", encoding="utf-8") as stream:
        for result in result_set:
            if result.string is not None:  # remove text of empty p tags
                stream.write(result.string + "\n")
