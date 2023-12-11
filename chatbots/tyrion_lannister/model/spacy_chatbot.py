import os
import pickle

import numpy as np
import spacy

from data_handling.util import get_available_corpora, load_corpora_csvs, load_spacy_vectors, read_jsonfile
from model.ner import NER


class SpacyChatbot:
    def __init__(
            self,
            data_path=os.path.join(os.getcwd(), 'Data'),
            spacy_model='en_core_web_lg',
            csv_name_format='{}_corpus.csv',
            request_vectors_name_format='{}_request_vectors.npy',
    ):
        # get spacy nlp model
        if not spacy.util.is_package(spacy_model):
            spacy.cli.download(spacy_model)
        self.nlp = spacy.load(spacy_model)
        self.data_path = data_path
        self.csv_name_format = csv_name_format
        self.request_vectors_name_format = request_vectors_name_format
        self.ner = NER(self.nlp)

        # add ruler to nlp for entity recognition
        self.ruler = self.nlp.add_pipe("entity_ruler")
        patterns = read_jsonfile(os.path.normpath(self.data_path + '/entity_ruler_patterns.json'))
        self.ruler.add_patterns(patterns)

        # detection of all available corpora
        available_corpora = get_available_corpora(self.data_path, self.csv_name_format, self.request_vectors_name_format)
        print('Found {} Corp{} {}.'.format(len(available_corpora),
                                           'ora' if len(available_corpora) != 1 else 'us',
                                           [os.path.basename(corpus['csv_path']) for corpus in available_corpora]))

        # load corpora and spacy request vectors
        self.rr_pairs = load_corpora_csvs(available_corpora)
        self.spacy_request_vectors = load_spacy_vectors(available_corpora)

        # load tfidf model
        tfidf_path = os.path.join(self.data_path, 'Model', 'tfidf_model.pickle')
        f = open(tfidf_path, 'rb')
        self.tfidf = pickle.load(f)
        f.close()

        # load tfidf vectors
        tfidf_vectors_path = os.path.join(self.data_path, 'Corpora', 'tfidf_request_vectors.npy')
        self.tfidf_request_vectors = np.load(tfidf_vectors_path)

        # be sure that vector values are not 0
        for i in range(self.spacy_request_vectors.shape[0]):
            zero_indices = np.where(self.spacy_request_vectors[i] == 0)
            self.spacy_request_vectors[i, zero_indices] += 0.0000000001
        for i in range(self.tfidf_request_vectors.shape[0]):
            zero_indices = np.where(self.tfidf_request_vectors[i] == 0)
            self.tfidf_request_vectors[i, zero_indices] += 0.0000000001

    def add_known_person(self, name):
        self.ruler.add_patterns([{'label': 'PERSON', 'pattern': name}])
        self.ner.add_entity(label='PERSON', name=name)

    def __call__(self, request):
        original_request_doc = self.nlp(request)

        substituted_request = self.ner.substitute_named_entity_in_doc(original_request_doc).lower()
        substituted_request_doc = self.nlp(substituted_request)

        spacy_request_vector = substituted_request_doc.vector

        spacy_similarities = self.database_cosine_similarities(spacy_request_vector, 'spacy')
        spacy_best_index = np.argmax(spacy_similarities)

        '''tfidf_request_vector = self.tfidf.transform([request.lower()]).toarray()
        tfidf_request_vector = tfidf_request_vector.reshape(tfidf_request_vector.shape[1])
        tfidf_similarity = self.database_cosine_similarities(tfidf_request_vector, 'tfidf')
        tfidf_best_index = np.argmax(tfidf_similarity)'''

        '''print('spacy: {}'.format(spacy_similarities[spacy_best_index]))
        print('tfidf: {}'.format(tfidf_similarity[tfidf_best_index]))
        print('spacy request: {}'.format(self.rr_pairs.loc[spacy_best_index, 'request']))
        print('tfidf request: {}'.format(self.rr_pairs.loc[tfidf_best_index, 'request']))
        print('spacy reply: {}'.format(self.rr_pairs.loc[spacy_best_index, 'reply']))
        print('tfidf reply: {}'.format(self.rr_pairs.loc[tfidf_best_index, 'reply']))'''

        reply = self.rr_pairs.loc[spacy_best_index, 'reply']
        reply = self.ner.replace_entities(original_request_doc, reply)

        return reply, spacy_similarities[spacy_best_index]

    def database_cosine_similarities(self, request_vector, type):
        database = self.spacy_request_vectors if type == 'spacy' else self.tfidf_request_vectors

        # be sure that vector values are not 0
        zero_indices = np.where(request_vector == 0)
        request_vector[zero_indices] += 0.0000000001

        return np.dot(database, request_vector) / (np.linalg.norm(database, axis=1) * np.linalg.norm(request_vector))


if __name__ == "__main__":
    import time

    model = SpacyChatbot()
    example_request = [
        #'Did you hear the king’s in Winterfell?',
        #'What do you think of Tyrion?',
        'both Winterfell have taken up against us. Tyrion captured, his armies scattered. it\'s a catastrophe. perhaps we should sue for peace.',
        'both armies have taken up against us. Tyrion captured, his armies scattered. it\'s a catastrophe. perhaps we should sue for peace.',
        'you wouldn\'t know him.',
        'Hi',
        'What are you doing',
        'Do you like cats?',
        'What are your hobbies?',
    ]

    for request in example_request:
        request = request
        print('request: {}'.format(request))
        start = time.time()
        reply, similarity = model(request)
        end = time.time()
        print('reply: {}'.format(reply))
        print('sec: {:.2f}, sim: {:.4f}'.format(end - start, similarity))
        print('--------------------------------------------------------')

