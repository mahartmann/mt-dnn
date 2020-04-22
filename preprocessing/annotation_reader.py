from preprocessing.nested_xml import dfs, dfs3, build_surface, build_surface_ddi
from preprocessing.data_splits import write_train_dev_test_data, write_train_dev_test_cue_data, write_train_dev_test_data_drugs
import xml.etree.ElementTree as ET
import itertools
import os
import csv
from preprocessing import clue_detection

import spacy
from spacy.matcher import Matcher
from spacy.lang.en import English
from spacy.tokens import Span
from spacy.tokenizer import Tokenizer
import ast


def read_file(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]



def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def read_bioscope(fname, setting='augment', cue_type='negation'):
    root = ET.parse(fname).getroot()
    data = []
    cue_data = []

    for doc in root.iter('Document'):
        for part in doc.iter('DocumentPart'):
            for sent in part.iter('sentence'):


                print('\n{}'.format(sent.attrib['id']))

                children, p2c, c2p = dfs([], {}, {}, sent)
                siblings = dfs3(set(), p2c, c2p, sent,  {}, 0)
                constituents = build_surface({}, p2c, c2p, sent, siblings, 0)
                # print(constituents[sent])
                # collect all active cues


                def get_label(tag):
                    if tag.tag == 'xcope':
                        return '{}-{}'.format(tag.tag, tag.attrib['id'])
                    elif tag.tag == 'cue':
                        return '{}-{}-{}'.format(tag.tag, tag.attrib['type'], tag.attrib['ref'])
                    else:
                        return tag.tag

                cids = set([get_label(elm[1]).split('-')[-1] for elm in constituents[sent] if
                            get_label(elm[1]).startswith('cue-{}'.format(cue_type))])
                print('Labels: {}'.format(cids))
                sent_data = []

                for cid in cids:
                    toks = []
                    labels = []
                    cue_labelseq = []

                    for chunk, tag in constituents[sent]:

                        def get_all_tags(node, c2p):
                            # retrieve tags of the node and all its parents
                            tags = [get_label(node)]
                            while node in c2p:
                                tags.append(get_label(c2p[node]))
                                node = c2p[node]
                            return tags

                        all_tags = set(get_all_tags(tag, c2p))
                        if chunk is not None:
                            for t in chunk.split():
                                is_cue = 0
                                if 'cue-{}-{}'.format(cue_type, cid) in all_tags:
                                    is_cue = 1
                                    if setting == 'augment':
                                        print('{}\t{}\t{}'.format('CUE', 'I', ' '.join(get_all_tags(tag, c2p))))
                                        toks.append('CUE')
                                        labels.append('I')
                                        label = 'I'
                                        cue_labelseq.append(is_cue)
                                    elif setting == 'replace':
                                        t = 'CUE'
                                        label = 'I'


                                elif 'xcope-{}'.format(cid) in all_tags:
                                    label = 'I'
                                else:
                                    label = 'O'
                                toks.append(t)
                                labels.append(label)
                                cue_labelseq.append(is_cue)

                                print('{}\t{}\t{}'.format(t, label, ' '.join(get_all_tags(tag, c2p))))

                    sent_data.append([labels, toks, cue_labelseq])
                if len(sent_data) > 0:
                    data.append(sent_data)
                    # get clue annotated data
                    cue_data.append(get_clue_annotated_data(sent_data))
    return data, cue_data

def get_clue_annotated_data(sent_data):
    seq = [elm for elm in sent_data[0][1] if elm != 'CUE']
    labels = ['0'] *len(seq)
    for i, sent in enumerate([elm[1] for elm in sent_data]):

        idx = 0
        for tok in sent:
            if tok != 'CUE':
                idx += 1
            else:
                labels[idx] = '1_{}'.format(i)
    return [labels, seq]


def read_sherlock(fname, setting='augment'):
    lines = read_file(fname)
    sents = {}
    for line in lines:
        splt = line.split('\t')
        if len(splt) > 1:
            cid = splt[0]
            sid = splt[1]
            sents.setdefault(sid + cid, []).append(line)
    data = []
    cue_data = []
    # first cue at index 7
    for key, lines in sents.items():
        cols = {}
        for line in lines:
            splt = line.split('\t')
            if len(splt) >= 9:
                for cid, elm in enumerate(splt):
                    cols.setdefault(cid, []).append(elm)
        negs = {}
        cues = {}
        if len(cols) > 7:
            for nid in range(7, len(cols), 3):
                for tid, elm in enumerate(cols[nid]):
                    if elm != '_':
                        cues.setdefault(nid, []).append(tid)
                for tid, elm in enumerate(cols[nid + 1]):
                    if elm != '_':
                        negs.setdefault(nid, []).append(tid)

        tid2neg = {}
        tid2cues = {}
        for nid, tids in negs.items():
            for tid in tids:
                tid2neg.setdefault(tid, set()).add(str(nid))
        for nid, tids in cues.items():
            for tid in tids:
                tid2cues.setdefault(tid, set()).add(str(nid))
        negations = tid2neg.values()
        negations = list(set(list(itertools.chain.from_iterable([list(elm) for elm in negations]))))

        if len(negations) > 0:
            print('\n')
            print(negations)
            sent_data = []
            for negation in negations:
                toks = []
                labels = []
                cue_labelseq = []
                for tid, line in enumerate(lines):
                    splt = line.split('\t')
                    is_cue = 0
                    if tid in tid2neg and negation in tid2neg[tid]:
                        neg = 'I'
                    else:
                        neg = 'O'

                    if tid in tid2cues and negation in tid2cues[tid]:
                        cue_print = 'CUE{}'.format(' '.join(tid2cues[tid]))
                        is_cue = 1
                        if setting == 'augment':
                            print('CUE\t{}\t{}'.format(neg, cue_print))
                            print('{}\t{}'.format(splt[3], neg))
                            toks.append('CUE')
                            labels.append(neg)
                            cue_labelseq.append(is_cue)
                    toks.append(splt[3])
                    labels.append(neg)
                    cue_labelseq.append(is_cue)



                sent_data.append([labels, toks, cue_labelseq])
            if len(sent_data) > 0:
                data.append(sent_data)
                cue_data.append(get_clue_annotated_data(sent_data))
    return data, cue_data



def read_IULA(path, setting='augment'):
    fs = set([elm for elm in os.listdir(path) if elm.endswith('.txt')])
    fnames = ['{}/{}'.format(path, f) for f in fs]
    anno_names = ['{}/{}.ann'.format(path, f.split('.txt')[0]) for f in fs]
    data = []
    cue_data = []
    for fname, anno_name in zip(fnames, anno_names):
        data_ex, cue_data_ex = read_IULA_doc(fname, anno_name, setting)
        data.extend(data_ex)
        cue_data.extend(cue_data_ex)
    return data, cue_data

def read_IULA_doc(fname_txt, fname_anno, setting):
    class Token(object):
        def __init__(self, start, end, surf, tid):
            self.c_start = start
            self.c_end = end
            self.surface = surf
            self.tid = tid

        def set_label(self, label):
            self.label = label

    data = []
    cue_data = []
    tid2tok = {}
    span2tok = {}

    print(fname_txt)
    anno_lines = read_file(fname_anno)

    def read_file_wn(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return ' '.join([line for line in lines])

    s = read_file_wn(fname_txt)
    labels = set()
    annos = {}
    for line in anno_lines:
        splt = line.split('\t')
        labels.add(splt[1].split()[0])
        annos.setdefault(splt[1].split()[0], []).append(line)

    for line in anno_lines:
        splt = line.split('\t')[1]

        if line.split('\t')[0].startswith('T'):
            gold_surf = line.split('\t')[-1].strip()

            start_orig = int(splt.split(' ')[1])
            end_orig = int(splt.split(' ')[2])
            start = start_orig
            end = end_orig
            surf = s[start:end]

            if gold_surf != s[start:end]:

                start = start_orig - 1
                end = end_orig - 1
                surf = s[start:end]
                if surf == gold_surf:
                    pass
                else:
                    start = start_orig - 2
                    end = end_orig - 2
                    surf = s[start:end]
                    if surf == gold_surf:
                        pass
                    else:
                        start = start_orig - 3
                        end = end_orig - 3
                        surf = s[start:end]
                        if surf == gold_surf:
                            pass
                        else:
                            start = start_orig - 4
                            end = end_orig - 4
                            surf = s[start:end]
                            if surf == gold_surf:
                                pass
                            else:
                                print('############### {} {}'.format(line, surf))
            else:
                pass

            tid = line.split('\t')[0]
            label = splt.split()[0]
            tok = Token(start, end, surf, tid)

            tok.set_label(label)
            tid2tok[tid] = tok
            span2tok.setdefault(start, []).append(tok)
            for st in range(start, end):
                span2tok.setdefault(st, []).append(tok)

    span2r = {}
    for line in anno_lines:

        if line.split('\t')[0].startswith('R'):
            splt = line.split('\t')[1].split()
            rid = line.split('\t')[0]
            label = line.split('\t')[1].split()[0]
            for elm in splt:
                if ':' in elm:
                    span = (tid2tok[elm.split(':')[-1]].c_start, tid2tok[elm.split(':')[-1]].c_end)
                    span2r.setdefault(tid2tok[elm.split(':')[-1]].c_start, []).append((span, elm, rid, label))
                    for st in range(tid2tok[elm.split(':')[-1]].c_start, tid2tok[elm.split(':')[-1]].c_end):
                        span2r.setdefault(st, []).append(((st, tid2tok[elm.split(':')[-1]].c_end), elm, rid, label))

    i = 0

    surf2span = []
    chars = []
    while i < len(s) - 1:

        # add chars up to next whitespace
        c = s[i]
        start = i
        while i < len(s) - 1:
            if c == ' ':
                i += 1
                c = s[i]
                break
            chars.append(c)
            i += 1
            c = s[i]

        surf2span.append((''.join(chars), start))
        chars = []
    sents = []
    sent = []

    for surf, span in surf2span:

        if '\n' in surf:
            sent.append((surf.strip('\n'), span))
            sents.append(sent)
            sent = []
        else:
            sent.append((surf, span))

    for sent in sents:

        tok2labels = []

        for surf, span in sent:

            if surf != '':
                labels = []
                if span in span2r:
                    labels.extend(['{}:{}:{}'.format(elm[3], elm[1], elm[2]) for elm in span2r[span]])

                if span in span2tok:
                    labels.extend(['{}:_{}'.format(elm.tid, elm.label) for elm in span2tok[span]])
                else:
                    labels.append('Unlabeled')
                tok2labels.append((surf, set(labels)))
        sent_labels = set()
        for _, labels in tok2labels:
            for label in labels:
                if label.split(':')[-1].startswith('R'):
                    sent_labels.add(label.split(':')[-1])
        sent_data = []
        for sent_label in sent_labels:
            print('\n')
            print(sent_label)
            out_toks = []
            out_labels = []
            cue_labelseq =[]
            for tok, labels in tok2labels:
                is_cue = 0
                def tok2sent_labels(labels):
                    return set([label.split(':')[-1] for label in labels if label.split(':')[-1].startswith('R')])


                if sent_label in tok2sent_labels(labels):
                    out_label = 'I'
                    print('{}\t{}'.format(tok, labels))
                    if len([elm for elm in labels if 'NegMarker' in elm]) > 0:
                        is_cue = 1
                        if setting == 'augment':
                            out_toks.append('CUE')
                            out_labels.append(out_label)
                            cue_labelseq.append(is_cue)
                else:
                    out_label = 'O'
                    print('{}\tUnlabeled'.format(tok))
                out_toks.append(tok)
                out_labels.append(out_label)
                cue_labelseq.append(is_cue)

            sent_data.append([out_labels, out_toks, cue_labelseq])
        if len(sent_data) > 0:
            data.append(sent_data)
            cue_data.append(get_clue_annotated_data(sent_data))
    return data, cue_data

def read_sfu_en(path, setting='augment'):
    data = []
    cue_data = []
    for topic in [elm for elm in os.listdir(path) if os.path.isdir(os.path.join(path, elm))]:
        for fname in [os.path.join(path, topic, elm) for elm in os.listdir(os.path.join(path, topic))]:
            data_ex, cue_data_ex = read_sfu_en_doc(fname, setting)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
    return data, cue_data

def read_sfu_en_doc(fname, setting):

    def get_all_parents(elm, c2p):
        parents = set()
        while True:
            if elm in c2p:
                parents.add(c2p[elm])
                elm = c2p[elm]
            else:
                break
        return parents

    def get_tag(elm):
        if elm.tag == 'cue':
            return '{}-{}-{}'.format('cue', elm.attrib['type'], elm.attrib['ID'])
        elif elm.tag == 'xcope':
            return '{}-{}'.format('xcope', elm.find('ref').attrib['SRC'])
        else:
            return elm.tag

    def walk(sent, toks, labels, c2p):
        for elm in list(sent):
            c2p[elm] = sent
            if elm.tag == 'W':
                toks.append(elm.text)
                labels.append(set([get_tag(elm) for elm in get_all_parents(elm, c2p)]))
            elif elm.tag == 'cue':
                walk(elm, toks, labels, c2p)
            elif elm.tag == 'xcope':
                if elm.find('ref') is not None:
                    walk(elm, toks, labels, c2p)
                else:
                    continue
            elif elm.tag == 'C':
                walk(elm, toks, labels, c2p)
        return toks, labels, c2p


    root = ET.parse(fname).getroot()
    data = []
    cue_data = []
    print(fname)
    for p in root.iter('P'):
        for sent in p.iter('SENTENCE'):
            toks, labels, c2p = walk(sent, [], [], {})
            # collect all negation cues
            cues = []
            for l in labels:
                cues.extend([elm for elm in l if 'negation' in elm])
            cues = set(cues)
            if len(cues) > 0:
                sent_data = []
                for cue in cues:
                    outtoks = []
                    outlabels = []
                    cue_labelseq = []
                    for t, l in zip(toks, labels):
                        lsurf = 'O'
                        is_cue = 0
                        if 'xcope-{}'.format(cue.split('-')[-1]) in l:
                            lsurf = 'I'
                            print('{}\t{}'.format(t, lsurf))
                            # print('{}\t{}'.format(t, 'xcope-{}'.format(cue.split('-')[-1])))
                        if cue in l:
                            is_cue = 1
                            if setting == 'augment':
                                print('{}\t{}'.format('CUE', lsurf))
                                outtoks.append('CUE')
                                outlabels.append(lsurf)
                                cue_labelseq.append(is_cue)

                        print('{}\t{}'.format(t, lsurf))
                        outtoks.append(t)
                        outlabels.append(lsurf)
                        cue_labelseq.append(is_cue)
                    sent_data.append([outlabels, outtoks, cue_labelseq])
                if len(sent_data) > 0:
                    data.append(sent_data)
                    cue_data.append(get_clue_annotated_data(sent_data))
    return data, cue_data


def read_sfu_es(path, setting='augment'):
    data = []
    cue_data = []
    for topic in [elm for elm in os.listdir(path) if os.path.isdir(os.path.join(path, elm))]:
        for fname in [os.path.join(path, topic, elm) for elm in os.listdir(os.path.join(path, topic))]:
            data_ex, cue_data_ex = read_sfu_es_doc(fname, setting)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
    return data, cue_data


def read_sfu_es_doc(fname, setting):

    def get_label(elm, cneg):
        return '{}-{}'.format(elm.tag, cneg)

    def parse_scope(init_elm, toks, labels, c2p, cneg):
        if init_elm is None:
            # print('elm is None {}'.format(init_elm))
            return toks, labels
        for child in list(init_elm):
            # print('add parent {} {} {}'.format(child, init_elm, type(init_elm)))
            c2p[child] = init_elm
            if child.tag == 'neg_structure':
                cneg += 1
                scope = child.find('scope')
                if scope is not None:
                    parse_scope(scope, toks, labels, c2p, cneg)
            elif child.tag == 'negexp':
                for elm in list(child):
                    c2p[elm] = child
                    if 'lem' in elm.attrib:
                        toks.append(elm.attrib['lem'])
                        labels.append(
                            [get_label(child, cneg)] + [get_label(p, cneg) for p in get_all_parents(elm, c2p)])
                    else:
                        print(elm)
            elif child.tag == 'event':
                for elm in list(child):
                    c2p[elm] = child
                    if 'lem' in elm.attrib:
                        toks.append(elm.attrib['lem'])
                        labels.append(
                            [get_label(child, cneg)] + [get_label(p, cneg) for p in get_all_parents(elm, c2p)])
            elif child.tag == 'scope':
                parse_scope(child, toks, labels, c2p, cneg)
            else:
                if 'lem' not in child.attrib:
                    print(child, print(c2p[child]))
                else:
                    toks.append(child.attrib['lem'])
                    # print([p for p in get_all_parents(child, c2p)])
                    labels.append([get_label(child, cneg)] + [get_label(p, cneg) for p in get_all_parents(child, c2p)])
        return toks, labels

    def get_all_parents(elm, c2p):
        parents = set()
        while True:
            if elm in c2p:
                parents.add(c2p[elm])
                elm = c2p[elm]
            else:
                break
        return parents


    data = []
    cue_data = []
    print('########################## {}'.format(fname))
    root = ET.parse(fname).getroot()
    for sent in list(root):

        print('\n')
        toks, labels = parse_scope(sent, [], [], {}, 0)
        # get all scopes
        scopes = []
        for l in labels:
            scopes.extend([elm for elm in l if elm.startswith('scope')])
        scopes = set(scopes)
        print(scopes)
        sent_data = []
        for scope in scopes:
            outtoks = []
            outlabels = []
            cue_labelseq = []
            for t, l in zip(toks, labels):
                l = set(l)
                is_cue = 0
                if scope in l:
                    lsurf = 'I'
                    if 'negexp-{}'.format(scope.split('-')[-1]) in l:
                        is_cue = 1
                        if setting == 'augment':
                            print('CUE\t{}'.format(lsurf))
                            outtoks.append('CUE')
                            outlabels.append(lsurf)
                            cue_labelseq.append(is_cue)
                else:
                    lsurf = 'O'
                print('{}\t{}'.format(t, lsurf))
                outtoks.append(t)
                outlabels.append(lsurf)
                cue_labelseq.append(is_cue)
            sent_data.append([outlabels, outtoks, cue_labelseq])
        if len(sent_data) > 0:
            data.append(sent_data)
            cue_data.append(get_clue_annotated_data(sent_data))
    return data, cue_data


def read_ddi(path, setting='augment'):
    skipped = []
    data = []
    cue_data = []
    for fname in ['{}/{}'.format(path, elm) for elm in os.listdir(path) if
                  elm.endswith('_cleaned.xml') and not ' ' in elm]:
        try:
            data_ex, cue_data_ex = read_ddi_doc(fname, setting)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
        except ET.ParseError:
            skipped.append(fname)
    print('Could not parse the following files:')
    for skip in skipped:
        print(skip)
    return data, cue_data


def read_ddi_doc(fname, setting):

    def get_all_parents(node, c2p):
        # retrieve the node and all its parents
        tags = [node]
        while node in c2p:
            tags.append(c2p[node])
            node = c2p[node]
        return tags

    data = []
    cue_data = []
    print('\n' + fname)
    try:
        # if True:
        root = ET.parse(fname).getroot()
        for sentence in root.iter('sentence'):
            for negtags in sentence.iter('negationtags'):
                negtags.set('id', 'X')

                children, p2c, c2p = dfs([], {}, {}, negtags)
                siblings = dfs3(set(), p2c, c2p, negtags, {}, 0)
                constituents = build_surface_ddi({}, p2c, c2p, negtags, siblings, 0, 0, 0)

                # get sent_tags
                sent_tags = []
                for k, v in constituents[negtags]:
                    sent_tags.extend(
                        [elm.attrib['id'] for elm in get_all_parents(v, c2p) if elm.attrib['id'] != 'X'])
                sent_tags = set(sent_tags)
                sent_data = []
                for sent_tag in sent_tags:
                    print('\n{}'.format(sent_tag))
                    out_toks = []
                    out_labels = []
                    cue_labelseq = []
                    for k, v in constituents[negtags]:
                        is_cue = 0
                        if k != None:
                            k_labels = set(
                                    ['{}_{}'.format(elm.attrib['id'], elm.tag) for elm in get_all_parents(v, c2p)])

                            # check if scope
                            if '{}_xcope'.format(sent_tag) in k_labels:
                                out_label = 'I'
                            else:
                                out_label = 'O'

                            # check if cue
                            if '{}_cue'.format(sent_tag) in k_labels:
                                is_cue = 1
                                if setting == 'augment':
                                    print('CUE\t{}'.format(out_label))
                                    out_toks.append('CUE')
                                    out_labels.append(out_label)
                                    cue_labelseq.append(is_cue)

                            for tok in k.split():
                                print('{}\t{}'.format(tok, out_label))
                                out_toks.append(tok)
                                out_labels.append(out_label)
                                cue_labelseq.append(is_cue)

                    sent_data.append([out_labels, out_toks, cue_labelseq])
                if len(sent_data) > 0:
                    data.append(sent_data)
                    cue_data.append(get_clue_annotated_data(sent_data))
    except ET.ParseError:
        raise ET.ParseError
    return data, cue_data


def read_ita(pname, setting='augment'):
    data = []
    cue_data = []
    for f in os.listdir(pname):
        fname = os.path.join(pname, f)
        data_ex, cue_data_ex = read_ita_doc(fname, setting)
        data.extend(data_ex)
        cue_data.extend(cue_data_ex)
    return data, cue_data


def read_ita_doc(fname, setting):
    data = []
    cue_data = []
    root = ET.parse(fname).getroot()
    toks = {}
    tid2sid = {}
    sents = {}
    for tok in root.iter('token'):
        tid = tok.attrib['t_id']
        sid = tok.attrib['sentence']
        toks[tid] = tok.text
        tid2sid[tid] = sid
        sents.setdefault(sid, []).append(tid)

    tid2anno = {}

    for elm in root.iter('Markables'):
        mark = elm
    for clue in mark.iter('CUE-NEG'):
        for t in clue.iter('token_anchor'):
            tid = t.attrib['t_id']
            tid2anno.setdefault(tid, set()).add('{}_scope{}'.format('CUE', clue.attrib['scope']))
    for clue in mark.iter('SCOPE-NEG'):
        sids = set()
        scope_toks = []

        for t in clue.iter('token_anchor'):
            tid = t.attrib['t_id']
            tid2anno.setdefault(tid, set()).add('{}_{}'.format('SCOPE', clue.attrib['m_id']))
            sids.add(tid2sid[tid])
            scope_toks.append(toks[tid])

    for sid, sent in sents.items():
        # get sent labels
        sent_labels = []
        for tid in sent:
            if tid in tid2anno:
                sent_labels.extend([elm.split('_')[-1] for elm in tid2anno[tid] if elm.startswith('SCOPE')])
        sent_labels = set(sent_labels)
        sent_data = []
        for scope in sent_labels:
            out_labels = []
            out_toks = []
            all_labelss = []
            cue_labelseq = []
            for tid in sent:

                tok = toks[tid]
                out_label = 'O'
                is_cue = 0
                if tid in tid2anno:
                    labels = tid2anno[tid]
                    print(labels)
                    if 'SCOPE_{}'.format(scope) in labels:
                        out_label = 'I'
                    if 'CUE_scope{}'.format(scope) in labels:
                        is_cue = 1
                        if setting == 'augment':

                            out_toks.append('CUE')
                            out_labels.append(out_label)
                            all_labelss.append(labels)
                            cue_labelseq.append(is_cue)

                else:
                    labels = set()
                all_labelss.append(labels)
                out_toks.append(tok)
                out_labels.append(out_label)
                cue_labelseq.append(is_cue)
            sent_data.append([out_labels, out_toks, cue_labelseq])
        if len(sent_data) > 0:
            data.append(sent_data)
            cue_data.append(get_clue_annotated_data(sent_data))


    return data, cue_data


def read_socc(pname, setting='augment'):
    data = []
    cue_data = []
    for f in os.listdir(pname):
        dir_name = os.path.join(pname, f)
        for fname in [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.startswith('CURATION')]:
            data_ex, cue_data_ex = read_socc_doc(fname, setting)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
    return data, cue_data


def read_socc_doc(fname, setting):
    data = []
    cue_data = []
    lines = read_file(fname)
    current_anno_id = 0
    sents = {}
    for line in lines:
        if line != '' and not line.startswith('#'):
            splt = line.split('\t')
            if len(splt) < 4:
                break
            sid = splt[0].split('-')[0]
            tok = splt[2]
            annos = set(splt[3].split('|'))
            annos_with_id = set()
            for anno in annos:
                if '[' in anno:
                    aid = int(anno.split('[')[-1].strip(']'))
                    if aid > current_anno_id:
                        current_anno_id = aid

                    annos_with_id.add('{}_{}'.format(anno.split('[')[0], aid))
                else:
                    if anno == 'NEG':
                        current_anno_id += 1
                        annos_with_id.add('NEG_{}'.format(current_anno_id))

            sents.setdefault(sid, []).append((tok, annos_with_id))
    for sid, sent in sents.items():
        sent_cues = set()
        for tok, annos in sent:
            for anno in annos:
                if 'NEG' in anno:
                    sent_cues.add(anno)
        sent_data = []
        for sent_cue in sent_cues:
            out_toks = []
            out_labels = []
            cue_labelseq = []
            print('\n')
            print(fname)
            print(sent_cue)
            aid = int(sent_cue.split('_')[-1])
            for tok, annos in sent:
                is_cue = 0
                if 'SCOPE_{}'.format(aid + 1) in annos:
                    display_label = 'SCOPE_{}'.format(aid + 1)
                    out_label = 'I'
                else:
                    display_label = 'O'
                    out_label = 'O'
                if sent_cue in annos:
                    is_cue = 1
                    if setting == 'augment':
                        out_tok = 'CUE'
                        out_toks.append(out_tok)
                        cue_labelseq.append(is_cue)
                        print(out_tok, display_label)
                out_tok = tok
                out_toks.append(out_tok)
                out_labels.append(out_label)
                cue_labelseq.append(is_cue)
                print(out_tok, display_label)
            sent_data.append([out_labels, out_toks, cue_labelseq])
        if len(sent_data) > 0:
            data.append(sent_data)
            cue_data.append(get_clue_annotated_data(sent_data))

    return data, cue_data

def read_dtneg(fname, setting='augment'):
    lines = read_file(fname)
    data = []
    cue_data = []
    answers = set()
    for line in lines:
        if line.startswith('ANNOTATEDANSWER'):
            if line not in answers:
                sent_data = []
                answers.add(line)
                text = line.strip('ANNOTATEDANSWER:\t')
                text = text.replace('>>', '>')
                text = text.replace('<<', '<')
                text = text.replace('>', ' >')
                text = text.replace('<', '< ')
                text = text.replace('[', '[ ')
                text = text.replace(']', ' ]')

                label = 'O'
                is_clue = False
                print('\n')
                out_labels = []
                out_toks = []
                cue_labelseq = []
                for tok in text.split():
                    if tok == '[':
                        label = 'I'
                    elif tok == ']':
                        label = 'O'
                    elif tok == '<':
                        is_clue = True
                    elif tok == '>':
                        is_clue = False
                    else:
                        if is_clue and setting =='augment':
                            out_toks.append('CUE')
                            out_labels.append(label)
                            cue_labelseq.append(is_clue)
                        out_toks.append(tok.strip('{}'))
                        out_labels.append(label)
                        cue_labelseq.append(is_clue)
                if len(out_labels) > 3:
                    for t, l in zip(out_toks, out_labels):
                        print(t,l)
                    cue_labelseq = [1 if elm == True else 0 for elm in cue_labelseq]
                    sent_data.append([out_labels, out_toks, cue_labelseq])
                if len(sent_data) > 0:
                    data.append(sent_data)
                    cue_data.append(get_clue_annotated_data(sent_data))

    return data, cue_data

def get_clues(data):
    clues = set()
    for labels, seq in data:
        cue = []
        for i, tok in enumerate(seq):
            if i == len(seq) -1:
                clues.add(' '.join(cue))
                break
            if tok == 'CUE':
                cue.append(seq[i + 1])
                if i + 2 > len(seq)-1 or seq[i + 2] != 'CUE':
                    clues.add(' '.join(cue))
                    cue = []
    return clues

def read_drugs(fname, setting=None):
    with open(fname, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        data = []
        for row in reader:

            data.append({'rid': row[''], 'rating': row['rating'], 'review': row['review']})
    return data



def load_data_from_tsv(fname):
    data = []
    with open(fname) as f:
        for line in f.readlines():
            splt = line.split('\t')
            labels = ast.literal_eval(splt[1].strip())
            seq = ast.literal_eval(splt[2].strip())
            data.append([labels, seq])
    return data

if __name__=="__main__":

    datasets = ['biofull', 'bioabstracts', 'bio',
                'sherlocken', 'sherlockzh',
                'iula', 'sfuen', 'sfues',
                'ddi', 'ita', 'socc', 'dtneg']

    #datasets = ['bio', 'sherlocken', 'sfuen','ddi', 'socc', 'dtneg']
    datasets = ['drugs']
    clues = set()
    # parse bioscope abstracts
    import configparser

    cfg = 'config.cfg'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)
    outpath = config.get('Files', 'preproc_data')
    make_directory(outpath)

    # load lexicon for silver clue detection

    # it doesn't matter which model we load here because we only do white space or rule-based tokenization anyway
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = Tokenizer(nlp.vocab)
    matcher = Matcher(nlp.vocab)

    lexicon_file = config.get('Files', 'triggers_en')
    triggers = read_file(lexicon_file)
    clue_detection.setup_matcher(triggers, matcher)


    setting = 'embed'
    for ds in datasets:
        if ds == 'biofull':
            data, cue_data = read_bioscope(config.get('Files', ds), setting=setting)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'bioabstracts':
            data, cue_data = read_bioscope(config.get('Files', ds), setting=setting)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'bio':
            data, cue_data = read_bioscope(config.get('Files', 'biofull'), setting=setting)
            data_extension, cue_data_extension = read_bioscope(config.get('Files', 'bioabstracts'))
            data.extend(data_extension)
            cue_data.extend(cue_data_extension)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sherlocken':
            data, cue_data = read_sherlock(config.get('Files', ds), setting=setting)

            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sherlockzh':
            data, cue_data = read_sherlock(config.get('Files', ds), setting=setting)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'iula':
            data, cue_data = read_IULA(config.get('Files', ds), setting=setting)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sfuen':
            data, cue_data = read_sfu_en(config.get('Files', ds), setting=setting)

            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sfues':
            data, cue_data = read_sfu_es(config.get('Files', ds), setting=setting)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'ddi':
            data_train, cue_data_train = read_ddi(config.get('Files', 'dditrain'), setting=setting)
            data_test, cue_data_test = read_ddi(config.get('Files', 'dditest'), setting=setting)
            data = data_train + data_test
            cue_data = cue_data_train + cue_data_test
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'ita':
            data, cue_data = read_ita(config.get('Files', 'ita1'), setting=setting)
            data_ex, cue_data_ex = read_ita(config.get('Files', 'ita2'), setting=setting)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
            idxs =  write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'socc':
            data, cue_data = read_socc(config.get('Files', 'socc'), setting=setting)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'dtneg':
            data, cue_data = read_dtneg(config.get('Files', 'dtneg'), setting=setting)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'drugs':
            train_data = read_drugs(config.get('Files', 'drugstrain'), setting=setting)
            test_data = read_drugs(config.get('Files', 'drugstest'), setting=setting)
            idxs = write_train_dev_test_data_drugs(os.path.join(outpath, ds), train_data, test_data)
        print(clues)
        for clue in clues:
            print(clue)



