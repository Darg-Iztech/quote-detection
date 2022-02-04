'''
Usage:
  quote_detection.py <model-name> -t <corpus-type> -c <corpus-path> -e <embedding-path>

Options:
  -t           Corpus type (either parc, rwg, or stop)
  -c           Path to corpus
  -e           Path to embeddings file

'''



import random
from docopt import docopt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np

from evaluate import evaluate, report
from progressify import progressify

class LSTMSeq2Seq(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,  n_labels, layers=2, bidirectional=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_labels = n_labels
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers, bidirectional=bidirectional, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, n_labels)
        
    def forward(self, x):
        x = self.lstm(x)[0]
        if isinstance(x, nn.utils.rnn.PackedSequence):
            data = self.linear(x.data)
            return nn.utils.rnn.PackedSequence(data, x.batch_sizes)
        else:
            x = self.linear(x)
            return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_x, d_k, d_v):
        super().__init__()
        self.d_x = d_x
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_x, d_k)
        self.w_k = nn.Linear(d_x, d_k)
        self.w_v = nn.Linear(d_x, d_v)

    def forward(self, x):
        # x: float[batch, sequence_length, d_x]
        Q = self.w_q(x)
        # Q: float[batch, sequence_length, d_k]
        K = self.w_k(x)
        # K: float[batch, sequence_length, d_k]
        V = self.w_v(x)
        # V: float[batch, sequence_length, d_v]
        logits = torch.bmm(K, V.permute(0, 2, 1)) / np.sqrt(self.d_k)
        # logits float[batch, sequence_length, sequence_length]
        return torch.bmm(torch.softmax(logits, dim=-1), V)
        # return float[batch, sequence_length, d_v]

class MultiHeadedAttentionLayer(nn.Module):
    def __init__(self, d_x, n_heads):
        super().__init__()
        assert d_x % n_heads == 0
        self.n_heads = n_heads
        self.heads = [
            SelfAttentionLayer(d_x, d_x // n_heads, d_x // n_heads)
            for _ in range(n_heads)
        ]

    def forward(self, x):
        # x: float[batch, sequence_length, d_x]
        return torch.cat([
            head(x) for head in self.heads
        ], dim=-1)


class TransformerLayer(nn.Module):
    def __init__(self, d_x, n_heads, activation=F.relu):
        super().__init__()
        self.d_x = d_x
        self.n_heads = n_heads
        self.activation = activation
        self.attention = MultiHeadedAttentionLayer(d_x, n_heads)
        self.linear = nn.Linear(d_x, d_x)
        self.ln1 = nn.LayerNorm([d_x])
        self.ln2 = nn.LayerNorm([d_x])
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: float[batch, sequence, d_x]
        x_attn = self.dropout(self.attention(x))
        # x_attn: float[batch, sequence, d_x]
        x = self.ln1(x + x_attn)
        x_ff = self.dropout(self.activation(self.linear(x)))
        x = self.ln2(x + x_ff)
        return x
        
        
class TransformerSeq2Seq(nn.Module):
    def __init__(self, d_x, n_layers, n_heads, d_out, pos_encode=False):
        super().__init__()
        self.d_x = d_x
        self.d_out = d_out
        self.n_heads = n_heads
        self.pos_encode = pos_encode
        self.layers = [TransformerLayer(d_x, n_heads) for _ in range(n_layers)]
        self.out = nn.Linear(d_x, d_out)
        self.dropout = nn.Dropout(p=0.1)

    def _pos_encode(self, x):
        # x: float[batch, sequence_length, n_dims]
        batch, sequence_length, n_dims = x.shape
        positions = torch.arange(sequence_length, dtype=torch.float)
        # positions: float[sequence_length]
        frequencies = 10000 ** (-torch.arange(n_dims/2, dtype=torch.float) / (n_dims/2))
        # frequencies: float[n_dims / 2]
        coss = torch.cos(torch.ger(positions, frequencies))
        sins = torch.sin(torch.ger(positions, frequencies))
        pes = torch.cat([coss, sins], -1)
        # pes: float[sequence_length, n_dims]
        return self.dropout(x + pes)
        
    def forward(self, x):
        was_packed = False
        if isinstance(x, nn.utils.rnn.PackedSequence):
            was_packed = True
            x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # x float:[batch, sequence_length, d_x]
        if self.pos_encode:
            x = self._pos_encode(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        if was_packed:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        return x

class ComposedSeq2Seq(nn.Module):
    def __init__(self, dim, lstm_layers=2, transformer_layers=6):
        super().__init__()
        self.lstm = LSTMSeq2Seq(dim, dim, dim, layers=lstm_layers)
        self.transformer = TransformerSeq2Seq(dim, transformer_layers, 8, dim)

    def forward(self, x):
        return self.lstm(self.transformer(x))
        
class IdentityLayer(nn.Module):
    def forward(self, x):
        return x

class LSTMCRF(nn.Module):
    # only supports packedsequences of length 1...
    def __init__(self, dim, n_tags, lstm_layers=2):
        super().__init__()
        self.lstm = LSTMSeq2Seq(dim, dim, dim, layers=lstm_layers)
        self.crf = CRFLayer(dim, n_tags)
        # hack so we can print these nicely
        self.transitions = self.crf.transitions

    def forward(self, x):
        x = self.lstm(x)
        if isinstance(x, nn.utils.rnn.PackedSequence):
            return self.crf(x.data)
        else:
            return self.crf(x)

    def neg_log_likelihood(self, x, tags):
        x = self.lstm(x)
        if isinstance(x, nn.utils.rnn.PackedSequence):
            return self.crf.neg_log_likelihood(x.data, tags.data)
        else:
            return self.crf.neg_log_likelihood(x, tags)


class QuoteDetectionModel(nn.Module):
    def __init__(
        self,
        n_features,
        dim=300,
        lam=1e-4,
        encoder=None,
        use_crf=False,
        transformer=False,
        sample_steps=1,
        label_scheme='BE',
        viterbi=False,
        pipeline=True,
    ):
        super().__init__()
        self.n_features = n_features
        self.dim = dim
        self.lam = lam
        self.use_crf = use_crf
        self.sample_steps = sample_steps
        self.label_scheme = label_scheme
        self.viterbi = viterbi
        self.pipeline = pipeline
        self.embedding = nn.EmbeddingBag(n_features, dim, mode='sum')
        self.roles = list(roles)
        if encoder is None:
            self.encoder = lambda x: x
        elif encoder == 'transformer':
            self.encoder = TransformerSeq2Seq(dim, 6, 10, dim, pos_encode=True)
        if use_crf:
            self.seq2seqs = nn.ModuleList()
            self.crf_encoders = nn.ModuleList()
            self.crfs = nn.ModuleList()
            for role in self.roles:
                if transformer:
                    self.seq2seqs.append(TransformerSeq2Seq(dim, 6, 10, dim, pos_encode=True))
                else:
                    self.seq2seqs.append(LSTMSeq2Seq(dim, dim, dim, layers=2, bidirectional=True))
                #self.crfs.append(CRFLayer(dim, 3))
                self.crf_encoders.append(nn.Linear(dim, 3))
                self.crfs.append(ConditionalRandomField(3))
        else:
            self.seq2seqs = nn.ModuleList()
            self.outputs = nn.ModuleList()
            for role in self.roles:
                if transformer:
                    self.seq2seqs.append(TransformerSeq2Seq(dim, 6, 10, dim, pos_encode=True))
                else:
                    self.seq2seqs.append(LSTMSeq2Seq(dim, dim, dim, layers=2, bidirectional=True))
                self.outputs.append(nn.Linear(dim, 3))
        self.tag_embeddings = Parameter(torch.Tensor(3, 3, dim))
        nn.init.normal_(self.tag_embeddings)

    def forward(self, x):
        # x: long[batch_size, sequence_length, bag_size]
        nn.init.zeros_(self.embedding.weight[0])
        batch_size, sequence_length, bag_size = x.shape
        embedded = self.embedding(x.view(-1, bag_size)).view(batch_size, sequence_length, self.dim)
        # embedded: float[batch_size, sequence_length, self.dim]
        embedded = self.encoder(embedded)
        prediction_embeddings = torch.zeros_like(embedded)
        if self.use_crf:
            for step in range(self.sample_steps):
                paths = []
                for i, (seq2seq, crf_encoder, crf) in enumerate(zip(self.seq2seqs, self.crf_encoders, self.crfs)):
                    res = seq2seq(embedded + prediction_embeddings)
                    # res: float[batch_size, sequence_length, self.dim]
                    sequence_length = res.shape[1]
                    path, score = crf.viterbi_tags(crf_encoder(res).cpu(), torch.tensor([sequence_length], dtype=torch.int64).cpu())[0]
                    path = torch.tensor([path])
                    if self.pipeline:
                        prediction_embeddings = prediction_embeddings + F.embedding(path, self.tag_embeddings[i])
                    paths.append(path)
            return torch.stack(paths, dim=1)
            # return float:[batch_size, len(self.roles), sequence_length]
        else:
            for step in range(self.sample_steps):
                paths = []
                for i, (seq2seq, output) in enumerate(zip(self.seq2seqs, self.outputs)):
                    res = seq2seq(embedded + prediction_embeddings)
                    # res: float[batch_size, sequence_length, self.dim]
                    logits = output(res)
                    # logits: float[batch_size, sequence_length, 3]
                    if self.viterbi:
                        #breakpoint()
                        probs = torch.softmax(logits, dim=-1)
                        path = self.viterbi_sequence(probs)
                    else:
                        path = torch.argmax(logits, dim=-1)
                    if self.pipeline:
                        prediction_embeddings = prediction_embeddings + F.embedding(path, self.tag_embeddings[i])
                    paths.append(path)
            return torch.stack(paths, dim=1)
            # return float:[batch_size, len(self.roles), sequence_length]
            
    def neg_log_likelihood(self, x, tags):
        # x: long[batch_size, sequence_length, bag_size]
        # tags: int[batch_size, len(self.roles), sequence_length]
        nn.init.zeros_(self.embedding.weight[0])
        batch_size, sequence_length, bag_size = x.shape
        embedded = self.embedding(x.view(-1, bag_size)).view(batch_size, sequence_length, self.dim)
        # embedded: float[batch_size, sequence_length, self.dim]
        embedded = self.encoder(embedded)
        nll = 0
        prediction_embeddings = torch.zeros_like(embedded)
        if self.use_crf:
            for step in range(self.sample_steps):
                for i, (seq2seq, crf_encoder, crf) in enumerate(zip(self.seq2seqs, self.crf_encoders, self.crfs)):
                    role_tags = tags[:,i,:]
                    res = seq2seq(embedded + prediction_embeddings)
                    nll -= torch.sum(crf(crf_encoder(res), role_tags))
                    if self.pipeline:
                        score, path = crf(res)
                        prediction_embeddings = prediction_embeddings + F.embedding(path, self.tag_embeddings[i])
            return nll
        else:
            for step in range(self.sample_steps):
                for i, (seq2seq, output) in enumerate(zip(self.seq2seqs, self.outputs)):
                    role_tags = tags[:,i,:]
                    # role_tags: long[batch_size, sequence_length]
                    res = seq2seq(embedded + prediction_embeddings)
                    # res: float[batch_size, sequence_length, self.dim]
                    logits = output(res)
                    # logits: float[batch_size, sequence_length, 3]
                    nll += F.cross_entropy(logits.view(-1, 3), role_tags.view(-1), reduction='sum')
                    path = torch.argmax(logits, dim=-1)
                    if self.pipeline:
                        if self.viterbi:
                            # It is too expensive to do the viterbi step during training I think
                            # instead, let's use gold-standard labels
                            # error propogation yada yada
                            prediction_embeddings = prediction_embeddings + F.embedding(role_tags, self.tag_embeddings[i])
                        else:
                            prediction_embeddings = prediction_embeddings + F.embedding(path, self.tag_embeddings[i])
            return nll

    def viterbi_sequence(self, probs):
        if self.label_scheme == "BE":
            #naive = [" BE"[m] for m in torch.argmax(probs, -1)]
            paths = torch.zeros(probs.shape[:-1], dtype=torch.long)
            inf = float('inf')
            # OIB probabilities
            for i, pi in enumerate(probs):
                state_lps = [(0, -inf, -inf)]
                for lp_, lpb, lpe in torch.log(pi):
                    state_lpb = max(state_lps[-1]) + lpb
                    state_lpi = max(state_lps[-1][1], state_lps[-1][2]) + lp_
                    state_lpo = max(
                        state_lps[-1][0] + lp_,
                        state_lps[-1][1] + lpe,
                        state_lps[-1][2] + lpe
                    )
                    state_lps.append((state_lpo, state_lpi, state_lpb))


                #construct the most likely BIO sequence from back to front
                bios = []
                outside = True
                for (lpo, lpi, lpb) in reversed(state_lps[1:]):
                    if outside:
                        if lpo > lpi and lpo > lpb:
                            bios.append('O')
                        elif lpb > lpo and lpb > lpi:
                            bios.append('B')
                        else:
                            bios.append('I')
                            outside = False
                    else:
                        if lpb > lpi:
                            bios.append('B')
                            outside = True
                        else:
                            bios.append('I')

                bios.reverse()

                outside = True
                for j, bio in enumerate(bios):
                    if bio == 'B':
                        #final_labels.append('B')
                        paths[i,j] = 1
                        outside = False
                    elif bio == 'I':
                        #final_labels.append(' ')
                        paths[i,j] = 0
                        assert not outside
                    elif bio == 'O':
                        if not outside:
                            #final_labels.append('E')
                            paths[i,j] = 2
                        else:
                            #final_labels.append(' ')
                            paths[i,j] = 0
                        outside = True
        return paths
        
def jointshuffle(l1, l2):
    print("shuffling...")
    zipped = list(zip(l1, l2))
    random.shuffle(zipped)
    c, d = zip(*zipped)
    print("done shuffling")
    return list(c), list(d)


def batchify(feats, labels, batch_size):
    feats, labels = jointshuffle(feats, labels)
    feats.sort(key=len)
    labels.sort(key=lambda lab:lab.shape[1])
    

def encode_feats(*corpora):
    seen_features = {None: 0}

    encoded_corpora = []
    for corpus in corpora:
        encoded_corpus = []
        for document in corpus:
            bag_size = max([len(token) for token in document])
            document_tensor = []
            for token in document:
                bag = []
                for feature in token:
                    if feature not in seen_features:
                        seen_features[feature] = len(seen_features)
                    bag.append(seen_features[feature])
                while len(bag) < bag_size:
                    bag.append(0)
                document_tensor.append(bag)
            document_tensor = torch.tensor(document_tensor, dtype=torch.long)
            encoded_corpus.append(document_tensor)
        encoded_corpora.append(encoded_corpus)
    return encoded_corpora, seen_features

def encode_labels(*corpora, scheme='BE'):
    if scheme == 'BE':
        tags = ' BE'
    elif scheme == 'BIO':
        tags = 'OIB'

    encoded_corpora = []
    for corpus in corpora:
        encoded_corpus = []
        for document in corpus:
            document_tensor = []
            for role in roles:
                role_tensor = []
                for token in document:
                    role_tensor.append(tags.index(token[role]))
                document_tensor.append(role_tensor)
            document_tensor = torch.tensor(document_tensor, dtype=torch.long)
            encoded_corpus.append(document_tensor)
        encoded_corpora.append(encoded_corpus)
    return encoded_corpora

def inject_pretrained_embeddings(model, embedding_path, feat_indices):
    print("loading pre-trained embeddings...")
    with open(embedding_path) as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            if ('word', word) in feat_indices:
                index = feat_indices[('word', word)]
                v = torch.Tensor([float(li) for li in line[1:]])
                model.embedding.weight.data[index] = v
            if ('lemma', word) in feat_indices:
                index = feat_indices[('lemma', word)]
                v = torch.Tensor([float(li) for li in reversed(line[1:])])
                model.embedding.weight.data[index] = v

    print("done!")
def eval(loss_func, feats, labels):
    with torch.no_grad():
        loss = 0
        for fi, li in zip(feats, progressify(labels, "Evaluating datum %%i / %d" % len(labels))):
            loss += loss_func(fi, li)
        return loss / len(feats)

def train(loss_func, optimizer, feats, labels, lamb=1e-4):
    feats, labels = jointshuffle(feats, labels)
    mean_loss = None
    def progressify_str(i, _):
        s = "training datum %d / %d." % (i, len(labels))
        if i > 0:
            s += " Mean training loss: %f" % mean_loss
        return s
    for fi, li in zip(feats, progressify(labels, progressify_str)):
        optimizer.zero_grad()
        f = fi.unsqueeze(0)
        loss = loss_func(f, li.unsqueeze(0))
        if mean_loss is None:
            mean_loss = loss
        else:
            mean_loss = .995 * mean_loss + .005 * loss
        
        # l2 regularization
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                loss += lamb * torch.sum(param ** 2)
        loss.backward()
        optimizer.step()

def predict(forward_func, feats):
    predictions = []
    with torch.no_grad():
        for datum in feats:
            batch = datum.unsqueeze(0)
            #batch_feats = torch.stack(feats[i:i+predict_batch_size])
            batch_predictions = forward_func(batch)
            for pi in batch_predictions:
                predictions.append(pi)
    return predictions

def train_loop(model, optimizer, train_feats, train_labels, dev_feats, dev_labels, gamma=0.75, callback=None):
    loss_func = model.neg_log_likelihood
    #breakpoint()
    running_average = -1
    epoch = 0
    while True:
        print("Epoch %d" % epoch)
        model.eval()
        dev_score = callback()
        #print("Dev loss: %f" % dev_loss)
        running_average = gamma * running_average + (1-gamma) * dev_score
        print("Running average: %f" % running_average)
        if dev_score < running_average:
            break
        model.train()
        train(loss_func, optimizer, train_feats, train_labels)
        epoch += 1
    

def get_ev(model, feats, raw_labels, eval_mode='exact'):
    print("Predicting spans...")
    predicted = predict(model, feats)
    predicted_processed = []
    for doc in predicted:
        doc_processed = []
        for i in range(len(doc[0])):
            token_processed = {}
            for r, role in enumerate(roles):
                if scheme == 'BE':
                    token_processed[role] = ' BE  '[doc[r][i]]
                elif scheme == 'BIO':
                    token_processed[role] = 'OIBOO'[doc[r][i]]
            doc_processed.append(token_processed)
        predicted_processed.append(doc_processed)
    # BUG HERE!! should say scheme=scheme
    return evaluate(predicted_processed, raw_labels, roles=roles, mode=eval_mode)




def run_model(
        raw_train_feats, raw_train_labels, raw_dev_feats, raw_dev_labels, raw_test_feats, raw_test_label
):
    (train_feats, dev_feats, test_feats), feat_indices = encode_feats(raw_train_feats, raw_dev_feats, raw_test_feats)
    train_labels, dev_labels, test_labels = encode_labels(raw_train_labels, raw_dev_labels, raw_test_labels)
    
    n_feats = len(feat_indices)

    model = QuoteDetectionModel(n_feats, use_crf=use_crf, sample_steps=1, label_scheme=scheme, viterbi=False, transformer=False, pipeline=False)

    if embedding_path is not None:
        inject_pretrained_embeddings(model, embedding_path, feat_indices)

    optimizer = optim.Adam(model.parameters())
    best_f1 = -1
    def training_callback():
        nonlocal best_f1
        if check_presence:
            ev = get_ev(model, dev_feats, raw_dev_labels, eval_mode='presence')
        else:
            ev = get_ev(model, dev_feats, raw_dev_labels)
        print(report(ev, roles=roles))
        f1 = 0
        if 'content' in ev:
            tp = ev['content']['tp']
            fp = ev['content']['fp']
            fn = ev['content']['fn']
        else:
            tp = 0
            fp = 0
            fn = 0
            for role in ev:
                tp += ev[role]['tp']
                fp += ev[role]['fp']
                fn += ev[role]['fn']
                
        if tp != 0:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 / (1/p + 1/r)
        if f1 > best_f1:
            print('Best model so far! Saving...')
            best_f1 = f1
            torch.save(model.state_dict(), model_path)
        return f1

    train_loop(model, optimizer, train_feats, train_labels, dev_feats, dev_labels, callback=training_callback)
    print("Loading best model...")
    model.load_state_dict(torch.load(model_path))
    print("Evaluating on test-set...")
    ev = get_ev(model, test_feats, raw_test_labels, eval_mode='exact')
    print(report(ev, roles=roles))
    if check_presence:
        ev_presence = get_ev(model, test_feats, raw_test_labels, eval_mode='presence')
        print('presence/absence:')
        print(report(ev_presence, roles=roles))
        return ev, ev_presence
    else:
        return ev
    

if __name__ == '__main__':
    arguments = docopt(__doc__)
    model_name = arguments['<model-name>']
    corpus_type = arguments['<corpus-type>']
    corpus_path = arguments['<corpus-path>']
    embedding_path = arguments['<embedding-path>']

    assert corpus_type in {'parc', 'stop', 'rwg'}

    xvalidate = (corpus_type in {'stop', 'rwg'})
    check_presence = (corpus_type == 'rwg')


    if corpus_type in {'rwg', 'stop'}:
        roles = [
            'direct',
            'indirect',
            'free_indirect',
            'reported'
        ]
    elif corpus_type == 'parc':
        roles = ['content']

    cuda = True
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    if corpus_type == 'rwg':
        from rwg2feat import corpus_feats_and_labels
    elif corpus_type == 'stop':
        from stop2feat import corpus_feats_and_labels
    elif corpus_type == 'parc':
        from parc2feat import corpus_feats_and_labels

    import sys
    import os

    use_crf = False
    scheme = 'BE'

    model_path = model_name + '.pkl'

    if xvalidate:
        raw_feats, raw_labels = corpus_feats_and_labels(corpus_path, label_scheme=scheme)
        i_dev = len(raw_feats) // 10
        i_train = 2 * i_dev
        raw_feats, raw_labels = jointshuffle(raw_feats, raw_labels)
        for step in range(10):
            print("Cross-validation step %d" % step)

            raw_test_feats = raw_feats[:i_dev]
            raw_dev_feats = raw_feats[i_dev:i_train]
            raw_train_feats = raw_feats[i_train:]

            raw_test_labels = raw_labels[:i_dev]
            raw_dev_labels = raw_labels[i_dev:i_train]
            raw_train_labels = raw_labels[i_train:]
            
            run_model(
                raw_train_feats, raw_train_labels, raw_dev_feats, raw_dev_labels, raw_test_feats, raw_test_labels,
            )
            # cycle
            raw_feats = raw_feats[i_dev:] + raw_feats[:i_dev]
            raw_labels = raw_labels[i_dev:] + raw_labels[:i_dev]
    else:

        print('loading training data')
        raw_train_feats, raw_train_labels = corpus_feats_and_labels(os.path.join(corpus_path, 'train'), label_scheme=scheme)
        print('loading dev data')
        raw_dev_feats, raw_dev_labels = corpus_feats_and_labels(os.path.join(corpus_path, 'dev'), label_scheme=scheme)
        print('loading test data')
        raw_test_feats, raw_test_labels = corpus_feats_and_labels(os.path.join(corpus_path, 'test'), label_scheme=scheme)

        (train_feats, dev_feats, test_feats), feat_indices = encode_feats(raw_train_feats, raw_dev_feats, raw_test_feats)

        n_feats = len(feat_indices)
        train_labels, dev_labels, test_labels = encode_labels(raw_train_labels, raw_dev_labels, raw_test_labels, scheme=scheme)

        model = QuoteDetectionModel(n_feats, use_crf=use_crf, sample_steps=1, label_scheme=scheme, viterbi=False, transformer=False, pipeline=False)

        if embedding_path is not None:
            inject_pretrained_embeddings(model, embedding_path, feat_indices)

        optimizer = optim.Adam(model.parameters())
        best_f1 = -1
        def training_callback():
            global best_f1
            if check_presence:
                ev = get_ev(model, dev_feats, raw_dev_labels, eval_mode='presence')
            else:
                ev = get_ev(model, dev_feats, raw_dev_labels)
            print(report(ev, roles=roles))
            f1 = 0
            if 'content' in ev:
                tp = ev['content']['tp']
                fp = ev['content']['fp']
                fn = ev['content']['fn']
            else:
                tp = 0
                fp = 0
                fn = 0
                for role in ev:
                    tp += ev[role]['tp']
                    fp += ev[role]['fp']
                    fn += ev[role]['fn']

            if tp != 0:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 / (1/p + 1/r)
            if f1 > best_f1:
                print('Best model so far! Saving...')
                best_f1 = f1
                torch.save(model.state_dict(), model_path)
            return f1

        train_loop(model, optimizer, train_feats, train_labels, dev_feats, dev_labels, callback=training_callback)
        print("Loading best model...")
        model.load_state_dict(torch.load(model_path))
        print("Evaluating on test-set...")
        ev = get_ev(model, test_feats, raw_test_labels)
        print(report(ev, roles=roles))    
