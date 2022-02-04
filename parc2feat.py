import sys
import os
import re
import xml.etree.ElementTree as ET

'''
Features implemented:
* normalized text
* lemma
* pos
* start and end of constituents
'''

features_to_use = {
    'word',
    'constituent',
    'lemma',
    'pos',
}

#roles = ['source', 'cue', 'content']
roles = ['content']

def leaf_words(node):
    if node.tag == 'WORD':
        return [node]
    leaves = []
    for child in node:
        leaves += leaf_words(child)
    return leaves

def words_spans(words, preserve_nested=False):
    open_spans = {}
    closed_spans = {}
    for i, word in enumerate(words):
        current_spans = set()
        for attribution in word:
            for attributionRole in attribution:
                if attributionRole.attrib['roleValue'] in roles:
                    span_id = (attributionRole.attrib['roleValue'], attribution.attrib['id'])
                    current_spans.add(span_id)
        # close any open spans that don't continue here
        still_open_spans = {}
        for span_id in open_spans:
            if span_id in current_spans:
                still_open_spans[span_id] = open_spans[span_id]
            else:
                closed_spans[span_id] = (open_spans[span_id], i)
        open_spans = still_open_spans
        # open any new spans
        for span_id in current_spans:
            if span_id not in open_spans:
                open_spans[span_id] = i
    # close any spans still open
    for span_id in open_spans:
        closed_spans[span_id] = (open_spans[span_id], i)

    # make a dict from span role to a list of spans
    role2spans = {}
    for (role, _), (left, right) in closed_spans.items():
        if role not in role2spans:
            role2spans[role] = []
        role2spans[role].append((left, right))

    # filter nested spans
    if not preserve_nested:
        for role in role2spans:
            spans = role2spans[role]
            filtered = []
            for i, (l1, r1) in enumerate(spans):
                for l2, r2 in spans[:i] + spans[i+1:]:
                    if l2 <= l1 <= r2 or l2 <= r1 <= r2:
                        break
                else:
                    filtered.append((l1, r1))
            role2spans[role] = filtered
    return role2spans


def document_labels(path, scheme='BE'):
    tree = ET.parse(path)
    root = tree.getroot()
    words = leaf_words(root)
    spans = words_spans(words)
    if scheme == 'BE':
        labels = []
        for i in range(len(words)):
            l = {}
            for role in roles:
                l[role] = ' '
            labels.append(l)
        for role in spans:
            for l, r in spans[role]:
                labels[l][role] = 'B'
                if r < len(labels) and labels[r][role] == ' ':
                    labels[r][role] = 'E'
        return labels
    elif scheme == 'BIO':
        labels = []
        for i in range(len(words)):
            l = {}
            for role in roles:
                l[role] = 'O'
            labels.append(l)
        for role in spans:
            for l, r in spans[role]:
                labels[l][role] = 'B'
                for i in range(l+1, r):
                    labels[i][role] = 'I'
        return labels


def document_features(path):
    tree = ET.parse(path)
    root = tree.getroot()
    feats = node_features(root)
    return feats

def node_features(node, const=[], const_start = [], const_end = []):
    if node.tag == 'WORD':
        wf = word_features(node)
        if 'constituent' in features_to_use:
            for constituent in const_start:
                wf.append(('constituent start', constituent))
            for constituent in const_end:
                wf.append(('constituent end', constituent))
            for constituent in const:
                wf.append(('constituent', constituent))
        # cue features
        '''
        is_cue = False
        for attribution in node:
            for attributionRole in attribution:
                if attributionRole.attrib['roleValue'] == 'cue':
                    is_cue = True
                    break
        wf.append(('cue', is_cue))       
        '''
        return [wf]

    child_features = []
    for i, child in enumerate(node):
        if i == 0:
            child_const_start = const_start + [node.tag]
        else:
            child_const_start = []
        if i == len(node) - 1:
            child_const_end = const_end + [node.tag]
        else:
            child_const_end = []
        child_features += node_features(child, const + [node.tag], child_const_start, child_const_end)
    return child_features
 
def word_features(node):
    wf = []
    if 'word' in features_to_use:
        wf.append(("word", normalize_text(node.attrib['text'])))
    if 'lemma' in features_to_use:
        wf.append(("lemma", normalize_text(node.attrib['lemma'])))
    if 'pos' in features_to_use:
        wf.append(("pos", node.attrib['pos']))
    return wf

dig = re.compile('[0-9]')
def normalize_text(s):
    #such a clean and beautiful interface...
    if s is None:
        return s
    s = s.lower()
    s = re.sub(dig, '0', s)
    s = s.replace("``", '"')
    s = s.replace("''", '"')
    return s

def corpus_feats_and_labels(path, label_scheme='BE'):
    instance_feats = []
    instance_labels = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.xml'):
                fpath = os.path.join(dirpath, filename)
                instance_feats.append(document_features(fpath))
                instance_labels.append(document_labels(fpath, label_scheme))
    return instance_feats, instance_labels
