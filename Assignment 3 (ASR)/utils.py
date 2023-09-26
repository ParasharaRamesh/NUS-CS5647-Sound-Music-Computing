import os
import sys
import json
import nltk
from nltk.corpus import cmudict

ls = os.listdir
jpath = os.path.join


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
        data = json.loads(data)
    return data


def save_json(data, path, sort=False):
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=4, sort_keys=sort, ensure_ascii=False))


def print_json(data):
    print(json.dumps(data, indent=4, ensure_ascii=False))


class PhonemeUtil:
    def __init__(self):
        # Load the CMU Pronouncing Dictionary
        self.pronouncing_dict = cmudict.dict()
        self.unknown = "<UNK>"

    def get_phonemes(self, word):
        """Returns the phoneme sequence of a word."""
        ''' YOUR CODE HERE '''
        # convert to lowercase
        word = str.lower(word)
        if word in self.pronouncing_dict:
            return self.pronouncing_dict[word][0]
        else:
            return ['<UNK>']

    def word_to_phoneme_sequence(self, sentence):
        """Converts a word sequence to a phoneme sequence."""
        ''' YOUR CODE HERE '''
        sequence = []
        tokenized_sentence = nltk.word_tokenize(sentence)
        for word in tokenized_sentence:
            sequence.extend(self.get_phonemes(word))
        # this is for the purposes of matching the task 1 test cases
        sequence_without_phoneme_numbers = self.drop_number_in_phoneme(sequence)
        return sequence_without_phoneme_numbers

    def drop_number_in_phoneme(self, sequence):
        sequence_without_numbers = []
        for phoneme in sequence:
            if phoneme == self.unknown:
                corrected_phoneme = phoneme
            else:
                corrected_phoneme = ""
                # iterate until you see a number
                for token in phoneme:
                    if token.isalpha():
                        corrected_phoneme += token
                    else:
                        break
            sequence_without_numbers.append(corrected_phoneme)
        return sequence_without_numbers


class PhonemeTokenizer:
    def __init__(self):
        self.tokens = [  # ''' Use this as vocabulary. Do not modify it. '''
            #     Phoneme Example Translation
            # ------- ------- -----------
            '<blank>',  # blank token for CTC decoding
            'AA',  # odd     AA D
            'AE',  # at	AE T
            'AH',  # hut	HH AH T
            'AO',  # ought	AO T
            'AW',  # cow	K AW
            'AY',  # hide	HH AY D
            'B',  # be	B IY
            'CH',  # cheese	CH IY Z
            'D',  # dee	D IY
            'DH',  # thee	DH IY
            'EH',  # Ed	EH D
            'ER',  # hurt	HH ER T
            'EY',  # ate	EY T
            'F',  # fee	F IY
            'G',  # green	G R IY N
            'HH',  # he	HH IY
            'IH',  # it	IH T
            'IY',  # eat	IY T
            'JH',  # gee	JH IY
            'K',  # key	K IY
            'L',  # lee	L IY
            'M',  # me	M IY
            'N',  # knee	N IY
            'NG',  # ping	P IH NG
            'OW',  # oat	OW T
            'OY',  # toy	T OY
            'P',  # pee	P IY
            'R',  # read	R IY D
            'S',  # sea	S IY
            'SH',  # she	SH IY
            'T',  # tea	T IY
            'TH',  # theta	TH EY T AH
            'UH',  # hood	HH UH D
            'UW',  # two	T UW
            'V',  # vee	V IY
            'W',  # we	W IY
            'Y',  # yield	Y IY L D
            'Z',  # zee	Z IY
            'ZH',  # seizure	S IY ZH ER
            '<UNK>',  # unknown words
        ]
        self.vocab = set(self.tokens)

        ''' YOUR CODE HERE '''
        # Using dictionary comprehension to initialize it
        self.token_to_id = {
            token: i for i, token in enumerate(self.tokens)
        }  # a dictionary, map phoneme string to id
        self.id_to_token = {
            i: token for i, token in enumerate(self.tokens)
        }  # a dictionary, map phoneme id to string

    def encode_seq(self, seq):
        '''
        seq: a list of strings, each string represent a phoneme
        Return a list of numbers. Each representing the corresponding id of that phoneme
        '''
        ''' YOUR CODE HERE '''
        return [self.token_to_id[phoneme] for phoneme in seq]

    def decode_seq(self, ids):
        '''
        ids: a list of integers, each representing a phoneme's id
        Return a list of strings. Each string represent the phoneme of that id.
        '''
        ''' YOUR CODE HERE '''
        return [self.id_to_token[id] for id in ids]

    def decode_seq_batch(self, batch_ids):
        '''
        Apply decode_seq to a batch of phoneme ids
        '''
        ''' YOUR CODE HERE '''
        return [self.decode_seq(batch) for batch in batch_ids]


if __name__ == '__main__':
    pass
