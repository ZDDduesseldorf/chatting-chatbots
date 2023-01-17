import csv
import sys
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Union

import en_core_web_lg
from config import CSV_QUOTECHAR, CSV_SEPERATOR, corpus_path
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.language import Language
from spacy.tokens import Doc
from thinc.types import Floats1d


@dataclass
class CorpusEntry:
    prior_message: str
    spacy_doc: Doc
    tfidf_vector: Union[Floats1d, None]
    barneys_message: str


class Corpus:
    def __init__(self) -> None:
        loading_tupel = self.load_corpus()
        self.corpus = loading_tupel[0]
        self.nlp = loading_tupel[1]
        self.vectorizer = loading_tupel[2]

    def load_corpus(self) -> Tuple[List[CorpusEntry], Language, TfidfVectorizer]:
        """Get conversation peaces from csv"""

        corpus: List[CorpusEntry] = []

        with open(corpus_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(
                csvfile, delimiter=CSV_SEPERATOR, quotechar=CSV_QUOTECHAR
            )
            nlp = en_core_web_lg.load()

            for prior_message, barneys_message in reader:
                spacy_doc = nlp(prior_message)
                corpus.append(
                    CorpusEntry(prior_message, spacy_doc, None, barneys_message)
                )

            prior_messages = list(map(lambda entry: entry.prior_message, corpus))
            tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
            tfidf_vectorizer = tfidf_vectorizer.fit(prior_messages)
            for entry in corpus:
                entry.tfidf_vector = tfidf_vectorizer.transform([entry.prior_message])

            return corpus, nlp, tfidf_vectorizer


if __name__ == "__main__":
    pass
