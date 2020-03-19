from preprocessing.nested_xml import dfs, dfs3, build_surface, build_surface_ddi
from preprocessing.data_splits import write_train_dev_test_data
import xml.etree.ElementTree as ET
import itertools
import os


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
    for doc in root.iter('Document'):
        for part in doc.iter('DocumentPart'):
            for sent in part.iter('sentence'):
                labels = []
                toks = []

                print('\n{}'.format(sent.attrib['id']))
                annos = []
                all_text = []
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
                for cid in cids:
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

                                if 'cue-{}-{}'.format(cue_type, cid) in all_tags:

                                    if setting == 'augment':
                                        print('{}\t{}\t{}'.format('CUE', 'I', ' '.join(get_all_tags(tag, c2p))))
                                        toks.append('CUE')
                                        labels.append('I')
                                        label = 'I'
                                    elif setting == 'replace':
                                        t = 'CUE'
                                        label = 'I'
                                elif 'xcope-{}'.format(cid) in all_tags:
                                    label = 'I'
                                else:
                                    label = 'O'
                                toks.append(t)
                                labels.append(label)
                                print('{}\t{}\t{}'.format(t, label, ' '.join(get_all_tags(tag, c2p))))

                    data.append([labels, toks])

    return data


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
            for negation in negations:
                toks = []
                labels = []
                for tid, line in enumerate(lines):
                    splt = line.split('\t')

                    if tid in tid2neg and negation in tid2neg[tid]:
                        neg = 'I'
                    else:
                        neg = 'O'

                    if tid in tid2cues and negation in tid2cues[tid]:
                        cue_long = 'CUE{}'.format(' '.join(tid2cues[tid]))
                        neg = 'C'
                    else:
                        cue_long = ''

                    if setting == 'augment':
                        if neg == 'C':
                            neg = 'O'
                            print('CUE\t{}\t{}'.format(neg, cue_long))
                            print('{}\t{}'.format(splt[3], neg))
                            toks.append('CUE')
                            labels.append(neg)
                            toks.append(splt[3])
                            labels.append(neg)
                        else:
                            print('{}\t{}'.format(splt[3], neg))
                            toks.append(splt[3])
                            labels.append(neg)
                    else:
                        print('{}\t{}\t{}'.format(splt[3], neg, cue_long))
                data.append([labels, toks])
    return data



def read_IULA(path, setting='augment'):
    fs = set([elm for elm in os.listdir(path) if elm.endswith('.txt')])
    fnames = ['{}/{}'.format(path, f) for f in fs]
    anno_names = ['{}/{}.ann'.format(path, f.split('.txt')[0]) for f in fs]
    data = []
    for fname, anno_name in zip(fnames, anno_names):
        data.extend(read_IULA_doc(fname, anno_name, setting))
    return data

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
        out_toks = []
        out_labels = []
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
        for sent_label in sent_labels:
            print('\n')
            print(sent_label)

            for tok, labels in tok2labels:

                def tok2sent_labels(labels):
                    return set([label.split(':')[-1] for label in labels if label.split(':')[-1].startswith('R')])


                if sent_label in tok2sent_labels(labels):
                    out_label = 'I'
                    print('{}\t{}'.format(tok, labels))
                    if len([elm for elm in labels if 'NegMarker' in elm]) > 0:
                        if setting == 'augment':
                            out_toks.append('CUE')
                            out_labels.append(out_label)

                    out_toks.append(tok)
                    out_labels.append(out_label)
                else:
                    print('{}\tUnlabeled'.format(tok))
                    out_toks.append(tok)
                    out_labels.append('O')
            data.append([out_labels, out_toks])
    return data

def read_sfu_en(path, setting='augment'):
    data = []
    for topic in [elm for elm in os.listdir(path) if os.path.isdir(os.path.join(path, elm))]:
        for fname in [os.path.join(path, topic, elm) for elm in os.listdir(os.path.join(path, topic))]:
            data.extend(read_sfu_en_doc(fname, setting))
    return data

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
                for cue in cues:
                    outtoks = []
                    outlabels = []
                    for t, l in zip(toks, labels):
                        lsurf = 'O'
                        if 'xcope-{}'.format(cue.split('-')[-1]) in l:
                            lsurf = 'I'
                            print('{}\t{}'.format(t, lsurf))
                            outtoks.append(t)
                            outlabels.append(lsurf)
                            # print('{}\t{}'.format(t, 'xcope-{}'.format(cue.split('-')[-1])))
                        elif cue in l:
                            if setting == 'augment':
                                print('{}\t{}'.format('CUE', lsurf))
                                outtoks.append('CUE')
                                outlabels.append(lsurf)
                            print('{}\t{}'.format(t, lsurf))
                            outtoks.append(t)
                            outlabels.append(lsurf)
                        else:
                            print('{}\t{}'.format(t, lsurf))
                            outtoks.append(t)
                            outlabels.append(lsurf)
                    data.append([outlabels, outtoks])
    return data


def read_sfu_es(path, setting='augment'):
    data = []
    for topic in [elm for elm in os.listdir(path) if os.path.isdir(os.path.join(path, elm))]:
        for fname in [os.path.join(path, topic, elm) for elm in os.listdir(os.path.join(path, topic))]:
            data.extend(read_sfu_es_doc(fname, setting))
    return data


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
    print('########################## {}'.format(fname))
    root = ET.parse(fname).getroot()
    for sent in list(root):
        outtoks = []
        outlabels = []
        print('\n')
        toks, labels = parse_scope(sent, [], [], {}, 0)
        # get all scopes
        scopes = []
        for l in labels:
            scopes.extend([elm for elm in l if elm.startswith('scope')])
        scopes = set(scopes)
        print(scopes)
        for scope in scopes:
            outtoks = []
            outlabels = []
            for t, l in zip(toks, labels):
                l = set(l)
                if scope in l:
                    lsurf = 'I'
                    if 'negexp-{}'.format(scope.split('-')[-1]) in l:
                        if setting == 'augment':
                            print('CUE\t{}'.format(lsurf))
                            outtoks.append('CUE')
                            outlabels.append(lsurf)
                        print('{}\t{}'.format(t, lsurf))
                        outtoks.append(t)
                        outlabels.append(lsurf)
                    else:
                        print('{}\t{}'.format(t, lsurf))
                        outtoks.append(t)
                        outlabels.append(lsurf)

                else:
                    lsurf = 'O'
                    print('{}\t{}'.format(t, lsurf))
                    outtoks.append(t)
                    outlabels.append(lsurf)
            data.append([outlabels, outtoks])
    return data


def read_ddi(path, setting='augment'):
    skipped = []
    data = []
    for fname in ['{}/{}'.format(path, elm) for elm in os.listdir(path) if
                  elm.endswith('_cleaned.xml') and not ' ' in elm]:
        try:
            data.extend(read_ddi_doc(fname, setting))
        except ET.ParseError:
            skipped.append(fname)
    print('Could not parse the following files:')
    for skip in skipped:
        print(skip)
    return data


def read_ddi_doc(fname, setting):

    def get_all_parents(node, c2p):
        # retrieve the node and all its parents
        tags = [node]
        while node in c2p:
            tags.append(c2p[node])
            node = c2p[node]
        return tags

    data = []
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
                for sent_tag in sent_tags:
                    print('\n{}'.format(sent_tag))
                    out_toks = []
                    out_labels = []

                    for k, v in constituents[negtags]:
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
                                if setting == 'augment':
                                    print('CUE\t{}'.format(out_label))
                                    out_toks.append('CUE')
                                    out_labels.append(out_label)

                            for tok in k.split():
                                print('{}\t{}'.format(tok, out_label))
                                out_toks.append(tok)
                                out_labels.append(out_label)

                    data.append([out_labels, out_toks])
    except ET.ParseError:
        raise ET.ParseError
    return data


def read_ita(pname, setting='augment'):
    data = []
    for f in os.listdir(pname):
        fname = os.path.join(pname, f)
        data.extend(read_ita_doc(fname, setting))
    return data


def read_ita_doc(fname, setting):
    data = []
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
            tid2anno.setdefault(tid, set()).add('{}_scope{}'.format('CUE', clue.attrib['m_id'], clue.attrib['scope']))
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

        for scope in sent_labels:
            out_labels = []
            out_toks = []
            all_labelss = []

            for tid in sent:

                tok = toks[tid]
                out_label = 'O'
                if tid in tid2anno:
                    labels = tid2anno[tid]
                    if 'SCOPE_{}'.format(scope) in labels:
                        out_label = 'I'
                    if 'CUE_scope{}'.format(scope) in labels:
                        if setting == 'augment':
                            out_toks.append('CUE')
                            out_labels.append(out_label)
                            all_labelss.append(labels)

                else:
                    labels = set()
                all_labelss.append(labels)
                out_toks.append(tok)
                out_labels.append(out_label)
            data.append([out_labels, out_toks])

            for t, l, a in zip(out_toks, out_labels, all_labelss):
                print('{}\t{}\t{}'.format(t, l, a))
    return data


if __name__=="__main__":

    datasets = ['biofull', 'bioabstracts', 'bio',
                'sherlocken', 'sherlockzh',
                'iula', 'sfuen', 'sfues',
                'ddi', 'ita']

    # parse bioscope abstracts
    import configparser

    cfg = 'config.cfg'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)
    outpath = os.path.join(config.get('Files', 'data'), 'formatted')
    make_directory(outpath)
    for ds in datasets:
        if ds == 'biofull':
            data = read_bioscope(config.get('Files', ds))
            write_train_dev_test_data(os.path.join(outpath, ds), data)
        elif ds == 'bioabstracts':
            data = read_bioscope(config.get('Files', ds))
            write_train_dev_test_data(os.path.join(outpath, ds), data)
        elif ds == 'bio':
            data = read_bioscope(config.get('Files', 'biofull'))
            data.extend(read_bioscope(config.get('Files', 'bioabstracts')))
            write_train_dev_test_data(os.path.join(outpath, ds), data)
        elif ds == 'sherlocken':
            data = read_sherlock(config.get('Files', ds))
            write_train_dev_test_data(os.path.join(outpath, ds), data)
        elif ds == 'sherlockzh':
            data = read_sherlock(config.get('Files', ds))
            write_train_dev_test_data(os.path.join(outpath, ds), data)
        elif ds == 'iula':
            data = read_IULA(config.get('Files', ds))
            write_train_dev_test_data(os.path.join(outpath, ds), data)
        elif ds == 'sfuen':
            data = read_sfu_en(config.get('Files', ds))
            write_train_dev_test_data(os.path.join(outpath, ds), data)
        elif ds == 'sfues':
            data = read_sfu_es(config.get('Files', ds))
            write_train_dev_test_data(os.path.join(outpath, ds), data)
        elif ds == 'ddi':
            data_train = read_ddi(config.get('Files', 'dditrain'))
            data_test = read_ddi(config.get('Files', 'dditest'))
        elif ds == 'ita':
            data = read_ita(config.get('Files', 'ita1'))
            data.extend(read_ita(config.get('Files', 'ita2')))
            write_train_dev_test_data(os.path.join(outpath, ds), data)



