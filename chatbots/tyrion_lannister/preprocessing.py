import argparse
import os
import pickle

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from data_handling.corpora_preprocessing import preprocessing_method_mapping
from data_handling.util import CorpusType, get_word_blacklist_regex, read_textfile, files_in_directory, \
    get_available_corpora, load_corpora_csvs, read_jsonfile


def train_tfidf_vectorizer(data_path, corpora_data_path, vectorizer_path, csv_name_format, request_vector_name_format):
    print('Train TFIDF Model and save TFIDF Corpora Vectors')

    rr_pairs = load_all_available_corpora(data_path, csv_name_format, request_vector_name_format)

    # load stopword list
    stop_word_file = os.path.join(data_path, 'stopword_list.txt')
    stop_words = [line.replace('\n', '') for line in read_textfile(stop_word_file)]
    stop_words = list(set(stop_words))

    # train vectorizer
    tfidf = TfidfVectorizer(min_df=5, max_df=0.2, stop_words=stop_words, max_features=1000)
    tfidf.fit_transform(rr_pairs.request + rr_pairs.reply)

    # generate and save tfidf vectors of all requests
    tfidf_request_vectors = tfidf.transform(rr_pairs.request)
    tfidf_request_vectors = tfidf_request_vectors.toarray().astype(np.float32)
    save_path = os.path.join(corpora_data_path, request_vector_name_format.format('tfidf'))
    np.save(save_path, tfidf_request_vectors)

    # save model
    os.makedirs(vectorizer_path, exist_ok=True)
    f = open(os.path.join(vectorizer_path, 'tfidf_model.pickle'), 'wb')
    pickle.dump(tfidf, f)
    f.close()


def load_all_available_corpora(data_path, csv_name_format, request_vector_name_format):
    available_corpora = get_available_corpora(data_path, csv_name_format, request_vector_name_format)
    print('Found {} Corp{} {}.'.format(len(available_corpora),
                                       'ora' if len(available_corpora) != 1 else 'us',
                                       [os.path.basename(corpus['csv_path']) for corpus in available_corpora]))
    return load_corpora_csvs(available_corpora)


if __name__ == "__main__":
    # setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=os.path.join(os.getcwd(), 'Data'), required=False,
                        help='Path to data dir.')
    parser.add_argument('--keep_prior_data', action='store_true',
                        help='Does not delete previously preprocessed data.')
    parser.add_argument('--use_preset', action='store_true', help='Preprocesses the Preset Corpora set by developer.')
    parser.add_argument('--with_all', action='store_true', help='Preprocess all types of corpora.')
    parser.add_argument('--with_got', action='store_true', help='Preprocess the got transcripts.')
    parser.add_argument('--with_cornell', action='store_true', help='Preprocess the cornell movie-dialogs corpus.')
    parser.add_argument('--with_parliament', action='store_true', help='Preprocess the Parliament Question Time Corpus.')
    parser.add_argument('--with_daily', action='store_true', help='Preprocess the Daily Dialogs Corpus.')
    args = parser.parse_args()

    # define necessary names
    corpora_path = os.path.join(args.data_path, 'Corpora')
    model_path = os.path.join(args.data_path, 'Model')
    csv_filename_format = '{}_corpus.csv'
    request_vector_filename_format = '{}_request_vectors.npy'
    word_blacklist_file = os.path.join(args.data_path, 'word_blacklist.txt')
    word_blacklist_file = word_blacklist_file if os.path.isfile(word_blacklist_file) else None

    # delete files from previous run
    if not args.keep_prior_data:
        files_patterns = [csv_filename_format.format('*'), request_vector_filename_format.format('*')]
        files = files_in_directory(corpora_path, file_patterns=files_patterns, recursive=False)
        for file in files:
            os.remove(file)

    # create necessary objects
    if not spacy.util.is_package("en_core_web_lg"):
        spacy.cli.download("en_core_web_lg")
    nlp_model = spacy.load("en_core_web_lg")
    ruler = nlp_model.add_pipe("entity_ruler")
    patterns = read_jsonfile(os.path.normpath(args.data_path + '/entity_ruler_patterns.json'))
    ruler.add_patterns(patterns)

    word_blacklist_regex = get_word_blacklist_regex(word_blacklist_file)

    # collect processes
    all_preprocessing_types = []
    if args.with_got or args.with_all:
        all_preprocessing_types.append(CorpusType.GoT)
    if args.with_cornell or args.with_all:
        all_preprocessing_types.append(CorpusType.Cornell)
    if args.with_parliament or args.with_all:
        all_preprocessing_types.append(CorpusType.Parliament)
    if args.with_daily or args.with_all:
        all_preprocessing_types.append(CorpusType.DailyDialogs)

    if args.use_preset:
        all_preprocessing_types = [CorpusType.GoT, CorpusType.Cornell, CorpusType.DailyDialogs]

    # convert to preprocess methods (callable)
    mapping = preprocessing_method_mapping()
    all_preprocessing_methods = [mapping[process_type] for process_type in all_preprocessing_types]

    # preprocess corpora
    for preprocess in all_preprocessing_methods:
        preprocess(corpora_path, nlp_model, word_blacklist_regex, csv_filename_format, request_vector_filename_format)

    # train tfidf model
    train_tfidf_vectorizer(args.data_path, corpora_path, model_path, csv_filename_format, request_vector_filename_format)
