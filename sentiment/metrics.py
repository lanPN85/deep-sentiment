def pure_metric(f):
    def decorator(*args):
        true = args[0]
        pred = args[1]
        if len(pred) == 0 or len(pred) != len(true):
            raise ValueError("Invalid data length")
        return f(*args)
    return decorator


def zero_safe(f):
    def decorator(*args):
        v1 = args[0]
        v2 = args[1]
        if v1 == 0 or v2 == 0:
            return 0
        return f(*args)
    return decorator


@pure_metric
def accuracy(true, pred):
    correct = 0.0
    for t, p in zip(true, pred):
        if t == p:
            correct += 1.0
    return correct / float(len(true))


@pure_metric
def precision(true, pred, label):
    correct = 0.0
    total = 0.0
    for t, p in zip(true, pred):
        if p == label:
            total += 1
            if t == p:
                correct += 1
    return correct / total


@pure_metric
def recall(true, pred, label):
    correct = 0.0
    total = 0.0
    for t, p in zip(true, pred):
        if t == label:
            total += 1
            if t == p:
                correct += 1
    return correct / total


@zero_safe
def f1_score(prec, rec):
    return (2 * prec * rec) / (prec + rec)
