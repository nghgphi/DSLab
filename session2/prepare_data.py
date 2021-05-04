from os.path import isfile, join
# from lib import *
import os

from nltk.stem.porter import PorterStemmer
import os
import re

with open(r'tf_idf\data\stop_word.txt') as f:
    stop_words = f.read().splitlines()
    f.close()

stemmer = PorterStemmer()

def collect_data_from(parent_dir, newsgroup_list):
    data = []
    for group_id, newsgroup in enumerate(newsgroup_list):
        label = group_id
        dir_path = os.path.join(parent_dir, newsgroup)

        files = [(filename, os.path.join(dir_path, filename))
                for filename in os.listdir(dir_path) if isfile(os.path.join(dir_path, filename))
                ]
        files.sort()

        for filename, filepath in files:
            with open(filepath) as f:
                text = f.read().lower()
                words = [stemmer.stem(word)
                        for word in re.split('\W+', text)
                        if word not in stop_words]
                content = ' '.join(words)
                assert len(content.splitlines()) == 1
                data.append(str(label) + '<ffff>' + filename + '<ffff>' + content)
    return data

if __name__ == '__main__':

    path = r'tf_idf/data'
    current_path = os.getcwd()
    current_path = os.path.join(current_path, path)

    dirs = []
    for dir_name in os.listdir(current_path):
        if not isfile(os.path.join(current_path, dir_name)):
            dirs.append(os.path.join(current_path, dir_name))

    train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])

    list_newgroups = [newsgroup for newsgroup in os.listdir(train_dir)]
    list_newgroups.sort()

    train_data = collect_data_from(
        parent_dir= train_dir, 
        newsgroup_list= list_newgroups
    )
    test_data = collect_data_from(
        parent_dir= test_dir,
        newsgroup_list= list_newgroups
    )
    full_data = train_data + test_data

    train_data_processed = 'tf_idf/data/train_data_processed.txt'
    test_data_processed = 'tf_idf/data/test_data_processed.txt'
    full_data_processed = 'tf_idf/data/full_data_processed.txt'

    
    with open(os.path.join(os.getcwd(), train_data_processed), 'w') as f:
        f.write('\n'.join(train_data))
    with open(os.path.join(os.getcwd(), test_data_processed), 'w') as f:
        f.write('\n'.join(test_data))
    with open(os.path.join(os.getcwd(), full_data_processed), 'w') as f:
        f.write('\n'.join(full_data))




    