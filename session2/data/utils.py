from collections import defaultdict
from typing import DefaultDict
from lib import *

def generate_vocabulary(data_path):
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size * 1. / df)
    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    corpus_size = len(lines)

    for line in lines:
        features = line.split('<ffff>')
        text = features[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1
    
    word_idfs = [(word, compute_idf(doc_freq, corpus_size))
                for word, doc_freq in zip(doc_count.keys(), doc_count.values())
                if doc_freq > 10 and not word.isdigit()]
    
    word_idfs.sort(key= lambda word_idfs: word_idfs[1])
    print('Vocabulary size : {}'.format(len(word_idfs)))

    with open('tf_idf/data/words_list.txt', 'w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in word_idfs]))

def get_tf_idf(data_path):
    with open(r'data\words_idfs.txt') as f:
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1]))
                    for line in f.read().splitlines()]
        word_IDs = dict([(word, index)
                        for index, (word, idf) in enumerate(words_idfs)])
        idfs = dict(words_idfs)
    
    with open(data_path) as f:
        documents = [
            (int(line.split('<ffff>')[0]),
            int(line.split('<ffff>')[1]),
            line.split('<ffff>')[2])
            for line in f.read().splitlines()
        ]
    data_tf_idf = []
    for document in documents:
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])
        words_tf_idfs = []
        sum_square = 0.0

        for word in word_set:
            term_freq = words.count(word)
            tf_idf_value = term_freq * 1. / max_term_freq * idfs[word]
            words_tf_idfs.append((word_IDs[word], tf_idf_value))
            sum_square += tf_idf_value ** 2
        words_tf_idf_normalized = [str(index) + ':' + str(tf_idf_value / np.sqrt(sum_square))
                                    for index, tf_idf_value in words_tf_idfs]
        sparse_rep = ' '.join(words_tf_idf_normalized)
        data_tf_idf.append((label, doc_id, sparse_rep))
            
    with open(r'data\test_tf_idf.txt', 'w') as f:
        for label, doc_id, sparse_rep in data_tf_idf:
            f.write(''.join(str(label) + '<fff>' + str(doc_id) + '<fff>' + sparse_rep)) 
            f.write('\n')
# generate_vocabulary(r'tf_idf\data\full_data_processed.txt')
get_tf_idf(r'data\test_data_processed.txt')
