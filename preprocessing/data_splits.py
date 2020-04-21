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

def generate_splits(num_data):
    np.random.seed(42)
    # generate 70/15/15 splits
    idx = [i for i in range(num_data)]
    np.random.shuffle(idx)
    train_idxs = idx[:int(np.ceil(0.7*len(idx)))]
    dev_idxs = idx[int(np.ceil(0.7*len(idx))):int(np.ceil((0.7+0.15)*len(idx)))]
    test_idxs = idx[int(np.ceil((0.7+0.15)*len(idx))):]
    return train_idxs, dev_idxs, test_idxs

def write_train_dev_test_data(fstem, data, setting):
    split_idxs = generate_splits(len(data))
    for i, splt in enumerate(['train', 'dev', 'test']):
        idxs = split_idxs[i]
        out_data = []
        for idx in idxs:
            for elm in data[idx]:
                out_data.append(elm)
        number_elms = 2
        if len(elm[2]) > 0:
            number_elms = 3
        fstem_out = fstem
        if setting == 'embed':
            fstem_out = fstem+'embed'

        write_split('{}_{}.tsv'.format(fstem_out, splt), out_data, number_elms)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),  '{}_{}.tsv'.format(fstem_out, splt) ))
    return  split_idxs

def write_train_dev_test_cue_data(fstem, data, split_idxs):
    number_elms = 2
    for i, splt in enumerate(['train', 'dev', 'test']):
        out_data = []
        idxs = split_idxs[i]
        for idx in idxs:
            elm = data[idx]

            out_data.append([['I' if l.startswith('1') else 'O' for l in elm[0]  ], elm[1]])

        write_split('{}#cues_{}.tsv'.format(fstem, splt), out_data, number_elms)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),  '{}#cues_{}.tsv'.format(fstem, splt) ))
    return




def write_split(fname, data, num_of_elms):
    outlines = []
    if num_of_elms == 2:
        for l, t in data:
            outlines.append('{}\t{}\t{}\n'.format(len(outlines), l, t))
    elif num_of_elms == 3:
        for l, t, c in data:
            outlines.append('{}\t{}\t{}\t{}\n'.format(len(outlines), l, t, c))

    write_lines(fname, outlines)


if __name__=="__main__":
    np.random.seed(42)
    tr, d, te = generate_splits(100)
    print(tr)
    print(d)
    print(te)

    tr, d, te = generate_splits(100)
    print(tr)
    print(d)
    print(te)

    np.random.seed(42)
    tr, d, te = generate_splits(100)
    print(tr)
    print(d)
    print(te)


