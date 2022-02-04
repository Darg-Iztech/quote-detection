def _string_table(table, padding: int = 2) -> str:
    column_widths: Dict[int, int] = {}
    for row in table:
        if not isinstance(row, str):
            for i, col in enumerate(row):
                col = str(col)
                if i not in column_widths:
                    column_widths[i] = 0
                column_widths[i] = max(len(col), column_widths[i])
    s = ""
    for row in table:
        if isinstance(row, str):
            line = row
        else:
            line = ""
            for i, col in enumerate(row):
                col = str(col)
                full_width = column_widths[i] + padding
                line += col
                npad = full_width - len(col)
                line += " " * npad
            line = line.rstrip()
        s += line + "\n"
    return s

def report(ev, roles=['content', 'cue', 'source']):
    rows = []
    rows.append(["", "tp", "fp", "fn", "precision", "recall", "f1"])
    tp_tot = 0
    fp_tot = 0
    fn_tot = 0
    for role in roles:
        tp = ev[role]['tp']
        fp = ev[role]['fp']
        fn = ev[role]['fn']
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn
        if tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 / ((1/precision) + (1/recall))
        rows.append([
            role, tp, fp, fn, "%.2f" % (100*precision), "%.2f" % (100*recall), "%.2f" % (100*f1)
        ]) 
    tp = tp_tot
    fp = fp_tot
    fn = fn_tot
    if tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 / ((1/precision) + (1/recall))
    rows.append([
        'total', tp, fp, fn, "%.2f" % (100*precision), "%.2f" % (100*recall), "%.2f" % (100*f1)
    ])

    return _string_table(rows)
 
            

def evaluate(hypo_labels, gold_labels, scheme="BE", roles=["source", "content", "cue"], mode='exact'):
    def merge(ev1, ev2):
        if ev1 is None:
            return ev2
        for role in ev1:
            for c in ev1[role]:
                ev1[role][c] += ev2[role][c]
        return ev1

    ev = None

    for hypo_instance, gold_instance in zip(hypo_labels, gold_labels):
        ev2 = evaluate_instance(hypo_instance, gold_instance, scheme, roles=roles, mode=mode)
        ev = merge(ev, ev2)

    return ev

def evaluate_instance(hypo_labels, gold_labels, scheme="BE", roles=["source", "content", "cue"], mode='exact'):
    ev = {}
    for role in roles:
        ev[role] = {'tp': 0, 'fp': 0, 'fn': 0}

    if mode == 'exact':
        hypo_spans = spans_from_labels(hypo_labels, roles=roles)
        gold_spans = spans_from_labels(gold_labels, roles=roles)

        for role in roles:
            for span in hypo_spans[role]:
                if span in gold_spans[role]:
                    ev[role]['tp'] += 1
                else:
                    ev[role]['fp'] += 1
            for span in gold_spans[role]:
                if span not in hypo_spans[role]:
                    ev[role]['fn'] += 1
    elif mode == 'presence':
        for role in roles:
            hypo_present = False
            gold_present = False
            for label in hypo_labels:
                if label[role] not in ' O':
                    hypo_present = True
                    break
            for label in gold_labels:
                if label[role] not in ' O':
                    gold_present = True
                    break
            if hypo_present and gold_present:
                ev[role]['tp'] += 1
            elif hypo_present and not gold_present:
                ev[role]['fp'] += 1
            elif gold_present and not hypo_present:
                ev[role]['fn'] += 1

    return ev


def spans_from_labels(labels, scheme='BE', roles=['source', 'content', 'cue']):
    spans = {
        role: [] for role in roles
    }
    if scheme == 'BE':
        open_spans = {
            role: None for role in roles
        }
        for i, label in enumerate(labels):
            for role in label:
                if label[role] in 'BE':
                    if open_spans[role] is not None:
                        spans[role].append((open_spans[role], i))
                        open_spans[role] = None
                if label[role] == 'B':
                    open_spans[role] = i
        for role in open_spans:
            if open_spans[role] is not None:
                spans[role].append((open_spans[role], len(labels)))
        return spans
