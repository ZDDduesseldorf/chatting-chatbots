import os.path
import random
import re

from tqdm import tqdm

from data_handling.util import read_jsonfile, multi_replace


class NER:
    def __init__(self, nlp, pattern_path=os.path.normpath(os.getcwd() + '/Data/entity_ruler_patterns.json')):
        self.nlp = nlp
        patterns = read_jsonfile(pattern_path)
        self.entity_string_labels = list(set([pattern['label'] for pattern in patterns]))

        # get known entities from patterns
        self.known_entities = {label: [] for label in self.entity_string_labels}
        for pattern in patterns:
            if pattern['label'] in self.entity_string_labels:
                self.known_entities[pattern['label']].append(pattern['pattern'])

    def add_entity(self, label, name):
        if label not in self.entity_string_labels:
            raise ValueError('Unknown Label: {}. Valid Labels are: {}'.format(label, self.entity_string_labels))

        self.known_entities[label].append(name)

    def replace_entities(self, request_doc, reply):
        request_entities = [ent for ent in request_doc.ents if ent.label_ in self.entity_string_labels]
        replacement_list = [(ent.label_, ent.text) for ent in request_entities]
        for key, value in replacement_list:
            reply = reply.replace(key, value, 1)

        for label in self.entity_string_labels:
            while label in reply:
                random_entity = random.choice(self.known_entities[label])
                reply = reply.replace(label, random_entity, 1)

        return reply

    def substitute_named_entity_in_text(self, text):
        return self.substitute_named_entity_in_doc(self.nlp(text))

    def substitute_named_entity_in_doc(self, doc):
        entities = doc.ents
        entities = [ent for ent in entities if ent.label_ in self.entity_string_labels]

        if len(entities) == 0:
            return doc.text

        # replace entities
        entities_replacement = {re.escape(ent.text): ent.label_ for ent in entities}
        return multi_replace(doc.text, entities_replacement)

    def substitute_named_entities_in_rr_pairs(self, rr_pairs, name=''):
        rr_pairs.request = self.substitute_named_entities_in_series(rr_pairs.request, name, ' Request')
        rr_pairs.reply = self.substitute_named_entities_in_series(rr_pairs.reply, name, ' Reply')

        return rr_pairs

    def substitute_named_entities_in_series(self, series, corpus_name='', data_type_name=''):
        # unnecessary pipelines
        disabled_pipes = ['tagger', 'parser', 'textcat', 'textcat_multilabel', 'lemmatizer', 'trainable_lemmatizer',
                          'morphologizer', 'attribute_ruler', 'senter', 'sentencizer', 'tok2vec', 'transformer']

        # define pipeline
        pipeline = self.nlp.pipe(series, disable=disabled_pipes, n_process=4, batch_size=1)
        tqdm_desc = '{}: Substitute{} Entities'.format(corpus_name, data_type_name)
        pipeline = tqdm(pipeline, total=len(series), unit='Requests', desc=tqdm_desc)

        # apply pipeline
        for i, doc in enumerate(pipeline):
            entities = doc.ents
            entities = [ent for ent in entities if ent.label_ in self.entity_string_labels]

            if len(entities) == 0:
                continue

            # replace entities
            entities_replacement = {re.escape(ent.text): ent.label_ for ent in entities}
            series[i] = multi_replace(doc.text, entities_replacement)

        return series
