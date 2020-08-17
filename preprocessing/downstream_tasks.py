import csv
import json
from xml.etree import ElementTree as ET


def read_drugs(fname, setting=None):
    with open(fname, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        data = []
        for row in reader:
            review = row['review'].strip().replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').replace('&#039;', "'").strip('"')

            def convert_rating(rating):
                if rating >= 7:
                    return 1
                elif rating <= 4:
                    return -1
                else:
                    return 0
            rating = convert_rating(float(row['rating'].strip()))

            data.append({'rid': row[''].strip(), 'rating':rating, 'review': review})
    return data


def read_gad(fname, split):
    data = []
    with open(fname, encoding='utf-8') as f:
        if split == 'train':
            for line in f:
                splt = line.strip('\n').split('\t')
                data.append({'label': splt[1], 'seq': splt[0]})
        elif split == 'test':
            #skip first line
            f.readline()
            for line in f:
                splt = line.strip('\n').split('\t')
                data.append({'label': splt[2], 'seq': splt[1]})

    return data


def read_biorelex(fname):
    with open(fname) as f:
        data = json.load(f)
    out_data = []
    for elm in data:
        def parse_elm(elm):
            data = []
            text = elm['text']
            interactions = elm['interactions']
            for interaction in interactions:
                print(interaction)
                participants = interaction['participants']
                # collect the replacements for this relation
                replacements = {}
                # counter marks if a participant is the first, second, nth participant in the relation
                for counter, pid in enumerate(participants):
                    ent = elm['entities'][pid]
                    label = ent['label'].upper()
                    for name in ent['names'].keys():
                        for mention in ent['names'][name]['mentions']:
                            print(pid, mention, label)
                            start = mention[0]
                            end = mention[1]
                            replacements[(start, end, counter)] = label

                new_text = replace_mentions(replacements, text)
                print(new_text)
                print('\n')
                rel_data = {}
                rel_data['paperid'] = elm['paperid']
                rel_data['seq'] = new_text
                rel_data['label'] = interaction['label']
                rel_data['type'] = interaction['type']
                rel_data['implicit'] = interaction['implicit']
                data.append(rel_data)
            return data

        out_data.extend(parse_elm(elm))
        print('\n\n')
    return out_data


def replace_mentions(replacements, text):
    # produce a string where mentions of particpants are replaced by their labels
    spans = sorted(list(replacements.keys()), key=lambda x: x[1])
    new_text = ''
    for i, span in enumerate(spans):
        start = span[0]
        end = span[1]
        pid = span[2]
        if i == 0:
            new_text += text[:start] + '${}{}$'.format(replacements[span], pid)
        else:
            new_text += text[spans[i - 1][1]:start] + '${}{}$'.format(replacements[span], pid)
        if i == len(spans) - 1:
            new_text += text[end:]
    new_text = new_text.replace('  ', ' ')
    return new_text


def read_cdr(fname):
    root = ET.parse(fname).getroot()
    out_data = []
    for elm in root:
        for doc in elm.iter('document'):
            docid = doc.find('id').text
            entities = {}
            for passage in doc.iter('passage'):
                offset = int(passage.find('offset').text)
                text = passage.find('text').text
                # collect the mention annotations
                for rid, annotation in enumerate(passage.iter('annotation')):

                    process = True
                    aid = annotation.attrib['id']
                    # ignore if it's an individual mention in a composite role
                    for infon in annotation.iter('infon'):
                        if infon.attrib['key'] == 'CompositeRole':
                            if infon.text == 'IndividualMention':
                                process = False
                        elif infon.attrib['key'] == 'type':
                            atype = infon.text
                        elif infon.attrib['key'] == 'MESH':
                            mesh = infon.text
                    if process:
                        span_start = int(annotation.find('location').attrib['offset'])
                        span_end = span_start + int(annotation.find('location').attrib['length'])
                        ent_text = annotation.find('text').text
                        print('{}\t{}\t{}\t{}\t{}\t{}'.format(aid, atype, mesh, span_start, span_end, ent_text))
                        print(ent_text)
                        print(text[span_start - offset: span_end - offset])
                        assert ent_text == text[span_start - offset: span_end - offset]
                        entities.setdefault(mesh, []).append({'span': (span_start, span_end), 'type': type, 'text': ent_text})
            # iterate through relation annotations
            for relation in doc.iter('relation'):
                print('##########################################')
                rid = relation.attrib['id']
                for infon in relation.iter('infon'):
                    if infon.attrib['key'] == 'Chemical':
                        chemical = infon.text
                    elif infon.attrib['key'] == 'Disease':
                        disease = infon.text
                    elif infon.attrib['key'] == 'relation':
                        relation_type = infon.text
                # build a mention_replaced string for this relation
                print(entities[chemical])
                print(entities[disease])
                assert len(entities[chemical]) == 1
                assert len(entities[disease]) == 1
                span_chemical = (entities[chemical]['span'][0], entities[chemical]['span'][1], '')
                span_disease = (entities[disease]['span'][0], entities[disease]['span'][1], '')
                replacements = {span_chemical: 'CHEMICAL', span_disease: 'DISEASE'}
                new_text = replace_mentions(replacements, text)
                print(rid)
                print(new_text)

                #new_text = replace_mentions(replacements, text)
                #data_point = {}
                #data_point['seq'] = new_text


def read_ade_doc(fname_txt, fname_ann):
    #import scispacy
    import spacy
    nlp = spacy.load("en_core_web_sm")

    class Token(object):
        def __init__(self, start, end, surf, tid):
            self.c_start = start
            self.c_end = end
            self.surface = surf
            self.tid = tid

        def set_label(self, label):
            self.label = label

    with open(fname_txt) as f:
        text = f.read()
    f.close()
    doc = nlp(text)
    for sent in doc.sents:
        print(sent)
    with open(fname_ann) as f:
        annos = []
        for line in f:
            annos.append(line.strip())
    tid2tok = {}
    for anno in annos:
        splt = anno.split('\t')
        aid = splt[0]
        label = splt[1].split()[0]
        if aid.startswith('T'):
            surface = splt[2]
            if len(splt[1].split()) == 3:
                span_start = int(splt[1].split()[1])
                span_end = int(splt[1].split()[2].split(';')[0])
                text_span = text[span_start:span_end]
                tok = Token(span_start, span_end, surface, aid)
            elif len(splt[1].split()) == 4:
                span_start1 = int(splt[1].split()[1])
                span_end1 = int(splt[1].split()[2].split(';')[0])
                span_start2 = int(splt[1].split()[2].split(';')[1])
                span_end2 = int(splt[1].split()[3])
                if span_start2 - span_end1 > 3:
                    break
                text_span = ' '.join([text[span_start1:span_end1], text[span_start2:span_end2] ])
                if text_span != surface:
                    span_start2 -= 1
                    text_span = ' '.join([text[span_start1:span_end1], text[span_start2:span_end2]])
                if text_span != surface:
                    text_span = text_span.replace(' ', '')
                tok = Token(span_start1, span_end2, surface, aid)
            elif len(splt[1].split()) == 5:
                span_start1 = int(splt[1].split()[1])
                span_end1 = int(splt[1].split()[2].split(';')[0])
                span_start2 = int(splt[1].split()[2].split(';')[1])
                span_end2 = int(splt[1].split()[3].split(';')[0])
                span_start3 = int(splt[1].split()[3].split(';')[1])
                span_end3 = int(splt[1].split()[4])
                text_span = ' '.join([text[span_start1:span_end1], text[span_start2:span_end2], text[span_start3:span_end3]])
                if text_span != surface:
                    span_start2 -= 1
                    text_span = ' '.join([text[span_start1:span_end1], text[span_start2:span_end2]])
                if text_span != surface:
                    text_span = text_span.replace(' ', '')
                tok = Token(span_start1, span_end3, surface, aid)
            #print(anno)
            #print(len(surface))
            #print(len(text_span))
            assert surface.replace(' ', '') == text_span.replace(' ','')

            tid2tok[aid] = tok
    for anno in annos:
        splt = anno.split('\t')
        aid = splt[0]
        label = splt[1].split()[0]
        if aid.startswith('R'):
            arg1 = splt[1].split()[1].split(':')[1]
            arg2 = splt[1].split()[2].split(':')[1]
            span1 = (tid2tok[arg1].c_start, tid2tok[arg1].c_end, '')
            span2 = (tid2tok[arg2].c_start, tid2tok[arg2].c_end, '')
            replacements = {span1: tid2tok[arg1].surface, span2: tid2tok[arg2].surface}
            new_text = replace_mentions(replacements, text)
            #print(new_text)

def read_ddi_relations(fname):
    with open(fname, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        data = []
        for i, row in enumerate(reader):


            data.append({'uid': i, 'seq': row['sentence'], 'labels': row['label'], 'sid': row['index']})

        return data

def read_chemprot_relations(fname):
    with open(fname, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        data = []
        for i, row in enumerate(reader):

            print(row)
            data.append({'uid': i, 'seq': row['sentence'], 'labels': row['label'], 'sid': row['index']})
        return data

if __name__=="__main__":
    import configparser
    import os
    cfg = 'config.cfg'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)

    for split in ['train']:
        read_ddi_relations(os.path.join(config.get('Files', 'ddi_relations_path'), 'train.tsv'))
