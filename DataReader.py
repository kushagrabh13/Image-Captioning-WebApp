import json
import glob
import numpy as np
import pandas as pd
from PIL import Image
import pickle
    
def DataSet():

    token = 'data/Flickr8k/Flickr8k.token.txt'
    captions = open(token, 'r').read().strip().split('\n')
    d = {}
    for i, row in enumerate(captions):
        row = row.split('\t')
        row[0] = row[0][:len(row[0])-2]
        if row[0] in d:
            d[row[0]].append(row[1])
        else:
            d[row[0]] = [row[1]]

    
    train_images_file = 'data/Flickr8k/Flickr_8k.trainImages.txt'
    train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

    test_images_file = 'data/Flickr8k/Flickr_8k.testImages.txt'
    test_images = set(open(test_images_file, 'r').read().strip().split('\n'))
    
    # build vocabulary
    caps = []
    for key, val in d.items():
        for i in val:
            caps.append('<start> ' + i + ' <end>')

    words = [i.split() for i in caps]
    unique = []
    for i in words:
       unique.extend(i)

    unique = list(set(unique))
    unique = pickle.load(open('data/Flickr8k/unique.p', 'rb'))

    word2idx = {val:index for index, val in enumerate(unique)}
    idx2word = {index:val for index, val in enumerate(unique)}

    max_len = 0
    for c in caps:
        c = c.split()
        if len(c) > max_len:
            max_len = len(c)

    vocab_size = len(unique)

    return word2idx, idx2word, vocab_size, max_len

