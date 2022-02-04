import sys
import os
import xml.etree.ElementTree as ET

word_only = True

span_types = {
    'direct_speech': 'direct',
    'indirect_speech': 'indirect',
    'free_indirect_speech': 'free_indirect',
    'reported_speech': 'reported',
    'direct_thought': 'direct',
    'indirect_thought': 'indirect',
    'free_indirect_thought': 'free_indirect',
    'reported_thought': 'reported',
    'direct_writing': 'direct',
    'indirect_writing': 'indirect',
    'free_indirect_writing': 'free_indirect',
    'reported_writing': 'reported'
}

def document_features_and_labels(path, label_scheme='BE', split_sentences=True):
    with open(path) as f:
        s = f.read()
    root = ET.fromstring(s)
    text = ''
    for node in root[1]:
        if node.tail is not None:
            text += node.tail

    tokens = []
    spans = []
    label_set = set()
    char2tok = []
    for child in root:
        if child.tag == 'AnnotationSet':
            annotationSet = child.attrib.get('Name')
            if annotationSet == 'PreProc_Anno':
                for annotation in child:
                    label = annotation.attrib['Type']
                    start = int(annotation.attrib['StartNode'])
                    end = int(annotation.attrib['EndNode'])
                    if label == 'Token':
                        while len(char2tok) < end:
                            char2tok.append(-1)
                        for i in range(start, end):
                            char2tok[i] = len(tokens)

                        token_feats = [('word', text[start:end].lower())]
                        if not word_only:
                            for child in annotation:
                                if child.tag == 'Feature':
                                    feature_name = child[0].text
                                    feature_value = child[1].text
                                    if feature_name == 'featsRF' and feature_value is not None:
                                        for part in feature_value.split('.'):
                                            token_feats.append((feature_name, part))
                                    else:
                                        token_feats.append((feature_name, feature_value))
                        tokens.append(token_feats)
    # Tokens are out of order :/
    # re-order them here
    reordered_tokens = []
    reordered_char2tok = []
    last = -1

    for c in char2tok:
        if c != -1:
            if c != last:
                reordered_tokens.append(tokens[c])
            reordered_char2tok.append(len(reordered_tokens) - 1)
        else:
            reordered_char2tok.append(-1)
        last = c
    tokens = reordered_tokens
    char2tok = reordered_char2tok
    spans = []
    for child in root:
        if child.tag == 'AnnotationSet':
            annotationSet = child.attrib.get('Name')
            if annotationSet == 'RW_Anno':
                for annotation in child:
                    label = annotation.attrib['Type']
                    label_set.add(label)
                    if label not in span_types:
                        continue
                    start = int(annotation.attrib['StartNode'])
                    end = int(annotation.attrib['EndNode'])
                    while char2tok[start] == -1:
                        start += 1
                    end -= 1
                    while char2tok[end] == -1:
                        end -= 1
                    start_tok = char2tok[start]
                    end_tok = char2tok[end]+1
                    assert start_tok >= 0
                    assert end_tok > 0
                    spans.append((span_types[label], start_tok, end_tok))
    labels = []
    if label_scheme == 'BE':
        for token in tokens:
            labels.append({st: ' ' for st in span_types.values()})
        for span_type, start, end in spans:
                labels[start][span_type] = 'B'
                if end < len(labels):
                    labels[end][span_type] = 'E'
    if split_sentences:
        sentences_tokens = []
        sentences_labels = []
        for child in root:
            if child.tag == 'AnnotationSet':
                annotationSet = child.attrib.get('Name')
                if annotationSet == 'PreProc_Anno':
                    for annotation in child:
                        label = annotation.attrib['Type']
                        start = int(annotation.attrib['StartNode'])
                        end = int(annotation.attrib['EndNode'])
                        if label == 'Sentence':
                            while char2tok[start] == -1:
                                start += 1
                            end -= 1
                            while char2tok[end] == -1:
                                end -= 1
                            start_tok = char2tok[start]
                            end_tok = char2tok[end]+1

                            sentence_tokens = tokens[start_tok:end_tok]
                            sentence_labels = labels[start_tok:end_tok]
                            sentences_tokens.append(sentence_tokens)
                            sentences_labels.append(sentence_labels)

        return sentences_tokens, sentences_labels            
    else:
        return [tokens], [labels]

def corpus_feats_and_labels(path, label_scheme='BE'):
    instance_feats = []
    instance_labels = []
    if path.endswith('.xml'):
        i_f, i_l = document_features_and_labels(path)
        instance_feats += i_f
        instance_labels += i_l
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.xml'):
                fpath = os.path.join(dirpath, filename)
                i_f, i_l = document_features_and_labels(fpath)
                instance_feats += i_f
                instance_labels += i_l
    return instance_feats, instance_labels

