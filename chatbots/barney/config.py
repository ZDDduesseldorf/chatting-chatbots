import os

RESOURCES_DIRECTORY_NAME = "resources"
scraped_resources_folder_name = os.path.join(RESOURCES_DIRECTORY_NAME, "scraped")
processed_resources_folder_name = os.path.join(RESOURCES_DIRECTORY_NAME, "processed")
CORPUS_PATH_BARNEY = os.path.join(processed_resources_folder_name, "corpus.csv")
CORPUS_PATH_BARNEY_CHATGPT = os.path.join(
    RESOURCES_DIRECTORY_NAME, "movie-corpus", "barnisazedQA.csv"
)
CORPUS_PATH_JOE = os.path.join(
    RESOURCES_DIRECTORY_NAME, "movie-corpus", "fragen_antworten_joey.csv"
)

CSV_SEPERATOR = ","
CSV_QUOTECHAR = '"'
