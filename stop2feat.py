import sys
import re
from nltk import word_tokenize


label_mapping = {
    'NRS': 'reported',
    'NRW': 'reported',
    'NRT': 'reported',
    'NRSA': 'reported',
    'NRWA': 'reported',
    'NRTA': 'reported',
    'NRSAP': 'reported',
    'NRWAP': 'reported',
    'NRTAP': 'reported',
    'IS': 'indirect',
    'IW': 'indirect',
    'IT': 'indirect',
    'FIS': 'free_indirect',
    'FIW': 'free_indirect',
    'FIT': 'free_indirect',
    'DS': 'direct',
    'DW': 'direct',
    'DT': 'direct',
    'FDS': 'direct',
    'FDW': 'direct',
    'FDT': 'direct'
}

def corpus_feats_and_labels(path, label_scheme='BE', label_mapping=label_mapping):
    with open(path, 'rb') as f:
        lines = f.readlines()
    lines = [str(line, encoding='latin_1').strip() for line in lines]

    current_text = None
    current_spans = None
    open_span_start = None
    open_span_labels = None

    other_labels = {}

    feats = []
    labels = []
    
    for line in lines:
        if current_text is None:
            if line.startswith('<body>'):
                current_text = []
                current_spans = []
                open_span_start = None
                open_span_labels = None
        else:
            if not line.startswith('<'):
                for token in word_tokenize(line):
                    current_text.append([('word', token.lower())])
            elif line.startswith('</body'):
                if open_span_labels is not None:
                    for open_span_label in open_span_labels:
                        current_spans.append((open_span_label, open_span_start, len(current_text)))
                #print(current_text)
                feats.append(current_text)
                current_labels = [{label: ' ' for label in label_mapping.values()} for _ in current_text]
                for label, begin, end in current_spans:
                    current_labels[begin][label] = 'B'
                    if end < len(current_labels):
                        current_labels[end][label] = 'E'
                labels.append(current_labels)
                current_text = None
                current_spans = None
            elif line.startswith('<sptag'):
                attrs = {}
                kvs = line[1:-1].split()
                for kv in kvs[1:]:
                    try:
                        k, v = kv.split('=')
                        attrs[k] = v
                    except ValueError:
                        pass
                if 'level' in attrs:
                    # ignore nested spans for now
                    continue
                if open_span_labels is not None:
                    for open_span_label in open_span_labels:
                        current_spans.append((open_span_label, open_span_start, len(current_text)))
                raw_label = attrs['cat']
                if raw_label[0] == '#':
                    continue
                raw_labels = raw_label.split('-')
                labs = set()
                for raw_label in raw_labels:
                    #while raw_label[-1].islower():
                    #    raw_label = raw_label[:-1]
                    raw_label = ''.join([c for c in raw_label if c.isupper()])
                    if raw_label in label_mapping:
                        label = label_mapping[raw_label]
                        labs.add(label)
                    else:
                        if raw_label not in other_labels:
                            other_labels[raw_label] = 0
                        other_labels[raw_label] += 1
                if len(labels) > 0:
                    open_span_labels = labs
                    open_span_start = len(current_text)
                else:
                    open_span_labels = None
                    open_span_start = None
    return feats, labels
            


if __name__ == '__main__':
    path = sys.argv[1]
    f, l = corpus_feats_and_labels(path)
