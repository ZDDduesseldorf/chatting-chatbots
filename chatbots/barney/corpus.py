import csv
from dataclasses import dataclass
from typing import List, Tuple

import en_core_web_lg
from config import CSV_QUOTECHAR, CSV_SEPERATOR
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.language import Language
from spacy.tokens import Doc
from thinc.types import Floats1d


@dataclass
class CorpusEntryNoTfidf:
    prior_message: str
    spacy_doc: Doc
    barneys_message: str


@dataclass
class CorpusEntry:
    prior_message: str
    spacy_doc: Doc
    barneys_message: str
    tfidf_vector: Floats1d


class Corpus:
    def __init__(self, path: str, name: str) -> None:
        loading_tupel = self.load_corpus(path)
        self.corpus = loading_tupel[0]
        self.nlp = loading_tupel[1]
        self.vectorizer = loading_tupel[2]
        self.name = name

    def load_corpus(
        self, path: str
    ) -> Tuple[List[CorpusEntry], Language, TfidfVectorizer]:
        """Get conversation peaces from csv"""

        corpus_no_tfidf: List[CorpusEntryNoTfidf] = []

        nlp = en_core_web_lg.load()
        with open(path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(
                csvfile, delimiter=CSV_SEPERATOR, quotechar=CSV_QUOTECHAR
            )

            for prior_message, barneys_message in reader:
                spacy_doc = nlp(prior_message)
                corpus_no_tfidf.append(
                    CorpusEntryNoTfidf(prior_message, spacy_doc, barneys_message)
                )

        prior_messages = list(map(lambda entry: entry.prior_message, corpus_no_tfidf))
        tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
        tfidf_vectorizer = tfidf_vectorizer.fit(prior_messages)
        corpus: List[CorpusEntry] = []
        for entry in corpus_no_tfidf:
            tfidf_vector = tfidf_vectorizer.transform([entry.prior_message])
            corpus.append(
                CorpusEntry(
                    entry.prior_message,
                    entry.spacy_doc,
                    entry.barneys_message,
                    tfidf_vector,
                )
            )

        return corpus, nlp, tfidf_vectorizer
