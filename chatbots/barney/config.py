import os

RESOURCES_DIRECTORY_NAME = "resources"
scraped_resources_folder_name = os.path.join(RESOURCES_DIRECTORY_NAME, "scraped")
processed_resources_folder_name = os.path.join(RESOURCES_DIRECTORY_NAME, "processed")
corpus_path = os.path.join(processed_resources_folder_name, "corpus.csv")

CSV_SEPERATOR = ","
CSV_QUOTECHAR = '"'
