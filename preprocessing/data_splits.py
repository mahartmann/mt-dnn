import numpy as np
import os
"""
produce train/test/dev splits
"""

import random

def write_lines(fname, lines):
    with open(fname, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)
    f.close()

def generate_train_dev_test_splits(num_data):
    np.random.seed(42)
    # generate 70/15/15 splits
    idx = [i for i in range(num_data)]
    np.random.shuffle(idx)
    train_idxs = idx[:int(np.ceil(0.7*len(idx)))]
    dev_idxs = idx[int(np.ceil(0.7*len(idx))):int(np.ceil((0.7+0.15)*len(idx)))]
    test_idxs = idx[int(np.ceil((0.7+0.15)*len(idx))):]
    return train_idxs, dev_idxs, test_idxs

def generate_train_dev_splits(num_data):
    np.random.seed(42)
    # generate 85/15 splits
    idx = [i for i in range(num_data)]
    np.random.shuffle(idx)
    train_idxs = idx[:int(np.ceil(0.85*len(idx)))]
    dev_idxs = idx[int(np.ceil(0.85*len(idx))):]
    return train_idxs, dev_idxs


def write_train_dev_test_data(fstem, data, setting):
    split_idxs = generate_train_dev_test_splits(len(data))
    for i, splt in enumerate(['train', 'dev', 'test']):
        idxs = split_idxs[i]
        out_data = []
        for idx in idxs:
            for elm in data[idx]:
                out_data.append([len(out_data)] + elm)
        number_elms = 2
        if len(elm[2]) > 0:
            number_elms = 3
        fstem_out = fstem
        if setting == 'embed':
            fstem_out = fstem+'embed'

        write_split('{}_{}.tsv'.format(fstem_out, splt), out_data)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),  '{}_{}.tsv'.format(fstem_out, splt) ))
    return  split_idxs

def write_train_dev_test_cue_data(fstem, data, split_idxs):
    number_elms = 2
    for i, splt in enumerate(['train', 'dev', 'test']):
        out_data = []
        idxs = split_idxs[i]
        for idx in idxs:
            elm = data[idx]

            out_data.append([len(out_data), ['I' if l.startswith('1') else 'O' for l in elm[0]  ], elm[1]])

        write_split('{}#cues_{}.tsv'.format(fstem, splt), out_data)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),  '{}#cues_{}.tsv'.format(fstem, splt) ))
    return

def write_train_dev_test_data_drugs(fstem, train_data, test_data):
    """
      split train_data into train/dev, test_split is fixed
      :return:
    """
    split_idxs = generate_train_dev_splits(len(train_data))
    for i, splt in enumerate(['train', 'dev']):
        idxs = split_idxs[i]
        out_data = []
        for idx in idxs:
            out_data.append([len(out_data), train_data[idx]['rating'], train_data[idx]['review']])
        write_split('{}_{}.tsv'.format(fstem, splt), out_data)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),  '{}_{}.tsv'.format(fstem, splt) ))
    out_data = []
    splt = 'test'
    for elm in test_data:
        out_data.append([len(out_data), elm['rating'], elm['review']])
    write_split('{}_{}.tsv'.format(fstem, splt), out_data)
    print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(test_data), len(out_data),
                                                                           '{}_{}.tsv'.format(fstem, splt)))
    return split_idxs

def write_data_gad_format(fstem, train_data, test_data):
    """
    split train_data into train/dev, test_split is fixed
    :return:
    """
    split_idxs = generate_train_dev_splits(len(train_data))
    for i, splt in enumerate(['train', 'dev']):
        idxs = split_idxs[i]
        out_data = []
        for idx in idxs:
            out_data.append([len(out_data), train_data[idx]['label'], train_data[idx]['seq']])
        write_split('{}_{}.tsv'.format(fstem, splt), out_data)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),
                                                                           '{}_{}.tsv'.format(fstem, splt)))
    out_data = []
    splt = 'test'
    for elm in test_data:
        out_data.append([len(out_data), elm['label'], elm['seq']])
    write_split('{}_{}.tsv'.format(fstem, splt), out_data)
    print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(test_data), len(out_data),
                                                                       '{}_{}.tsv'.format(fstem, splt)))
    return split_idxs



def write_split(fname, data):
    outlines = []
    for elm in data:
        s = ''
        for f in elm:
            s += '{}\t'.format(f)
        outlines.append(s.strip('\t') + '\n')
    write_lines(fname, outlines)


if __name__=="__main__":
    np.random.seed(42)
    tr, d, te = generate_train_dev_test_splits(100)
    print(tr)
    print(d)
    print(te)

    tr, d, te = generate_train_dev_test_splits(100)
    print(tr)
    print(d)
    print(te)

    np.random.seed(42)
    tr, d, te = generate_train_dev_test_splits(100)
    print(tr)
    print(d)
    print(te)


