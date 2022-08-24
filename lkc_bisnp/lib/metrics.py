
import numpy as np
import pandas as pd
import math

from sklearn.base import clone
from sklearn.metrics._classification import multilabel_confusion_matrix, _check_zero_division, _check_set_wise_labels, _prf_divide, _warn_prf
from sklearn.utils.multiclass import unique_labels
import joblib as jlib

from lkc_bisnp.lib.utils import cerr


def precision_recall_mcc_support(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average=None,
    warn_for=("precision", "recall", "f-score"),
    sample_weight=None,
    zero_division="warn",
):

    _check_zero_division(zero_division)
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)

    # Calculate tp_sum, pred_sum, true_sum ###
    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=labels,
        samplewise=samplewise,
    )
    # see docs for confusion_matrix in binary classification
    tn_sum = MCM[:, 0, 0]
    fn_sum = MCM[:, 1, 0]
    tp_sum = MCM[:, 1, 1]
    fp_sum = MCM[:, 0, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division
    )
    recall = _prf_divide(
        tp_sum, true_sum, "recall", "true", average, warn_for, zero_division
    )

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == "warn" and ("f-score",) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(average, "true nor predicted", "F-score is", len(true_sum))

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0

    denom = np.sqrt((tp_sum + fp_sum) * (tp_sum + fn_sum) * (tn_sum + fp_sum) * (tn_sum + fn_sum))
    denom[denom == 0.0] = 1  # avoid division by 0
    mcc = (tp_sum * tn_sum - fp_sum * fn_sum) / denom

    # Average the results
    if average == "weighted":
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = np.float64(1.0)
            if zero_division in ["warn", 0]:
                zero_division_value = np.float64(0.0)
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            if pred_sum.sum() == 0:
                return (
                    zero_division_value,
                    zero_division_value,
                    zero_division_value,
                    None,
                )
            else:
                return (np.float64(0.0), zero_division_value, np.float64(0.0), None)

    elif average == "samples":
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != "binary" or len(precision) == 1
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        mcc = np.average(mcc, weights=weights)
        true_sum = None  # return no support

    return precision, recall, mcc, true_sum


def prepare_dataframe_metrics(y_true, y_pred, labels, fold_id):

    MCM = multilabel_confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    tn_sum = MCM[:, 0, 0]
    fn_sum = MCM[:, 1, 0]
    tp_sum = MCM[:, 1, 1]
    fp_sum = MCM[:, 0, 1]
    pp_sum = tp_sum + fp_sum
    pn_sum = fn_sum + tn_sum
    p_sum = tp_sum + fn_sum
    n_sum = fp_sum + tn_sum

    # MCC
    denom = np.sqrt((tp_sum + fp_sum) * (tp_sum + fn_sum) * (tn_sum + fp_sum) * (tn_sum + fn_sum))
    denom[denom == 0.0] = 1  # avoid division by 0
    mcc = (tp_sum * tn_sum - fp_sum * fn_sum) / denom

    return pd.DataFrame({'FOLD_ID': fold_id, 'CLASS': labels,
                         'TN': tn_sum, 'FN': fn_sum, 'TP': tp_sum, 'FP': fp_sum,
                         'PP': pp_sum, 'PN': pn_sum, 'P': p_sum, 'N': n_sum,
                         'MCC': mcc})


def fit_and_score(clf, X, y, train_idx, test_idx, fold_id):
    
    train_X = X[train_idx]
    train_y = y[train_idx]

    test_X = X[test_idx]
    test_y = y[test_idx]

    clf.fit(train_X, train_y)

    y_pred = clf.predict(test_X)

    labels = unique_labels(y)

    return prepare_dataframe_metrics(test_y, y_pred, labels=labels, fold_id=fold_id)


def cross_validate(clf, X, y, cv, n_jobs=1):

    # check suitability of X and y for k_fold
    X, y = prepare_stratified_samples(X, y, cv.get_n_splits() / cv.n_repeats)

    # iterating over split data, this can be parallelized

    cerr(f'[I - cross-validating with n_jobs={n_jobs}]')
    if n_jobs == 1:
        results = [fit_and_score(clone(clf), X, y, train, test, fold_id)
                   for fold_id, (train, test) in enumerate(cv.split(X, y))]

    else:
        from tqdm_joblib import tqdm_joblib, tqdm
        with tqdm_joblib(tqdm(desc='Cross-Validating', total=cv.get_n_splits())) as progress_bar:
            results = jlib.Parallel(n_jobs=n_jobs)(
                jlib.delayed(fit_and_score)(clone(clf), X, y, train, test, fold_id)
                for fold_id, (train, test) in enumerate(cv.split(X, y))
            )

    return pd.concat(results)


def cross_val_predict(clf, X, y, cv, n_jobs):

    # check suitability of X and y for k_fold
    X, y = prepare_stratified_samples(X, y, cv.get_n_splits() / cv.n_repeats)

    aggregate_preds = []
    from sklearn.model_selection import cross_val_predict as cvp
    splits = list(cv.split(X, y))
    kfold = cv.get_n_splits() // cv.n_repeats
    for i in range(0, len(splits), kfold):
        y_preds = cvp(clf, X, y, cv=splits[i:i + kfold], n_jobs=n_jobs)
        aggregate_preds.append(pd.DataFrame({'LABEL': y, 'PREDICTION': y_preds}))

    return pd.concat(aggregate_preds)


def prepare_stratified_samples(X, y, k_fold):
    """ check the suitability of sample sets and modify X and y accordingly """

    label_counts = []
    for label, count in zip(* np.unique(y, return_counts=True)):
        # we make sure that every group has at least 2 * k_fold member
        if count < k_fold * 2:
            label_counts.append((label, math.ceil(k_fold * 2 / count)))

    if len(label_counts) == 0:
        # nothing to modify
        return (X, y)

    cerr('[I - prepare_stratified_sample() replicated group: %s]'
         % ' '.join(x[0] for x in label_counts))

    #import IPython; IPython.embed()
    aggregate_X = [X]
    aggregate_y = [y]
    for label, m_factor in label_counts:
        indexes = np.where(y == label)
        for i in range(m_factor):
            aggregate_X.append(X[indexes])
            aggregate_y.append(y[indexes])

    X = np.concatenate(aggregate_X, axis=0)
    y = np.concatenate(aggregate_y, axis=0)

    return (X, y)


def create_missingness(X, proportion, seed=None, missing_value=np.nan):

    if proportion <= 0:
        return X

    item_len = X.shape[1]
    miss_len = round(item_len * proportion)
    miss_mask = np.array([True] * miss_len + [False] * (item_len - miss_len))

    miss_X = np.copy(X)

    rng = np.random.default_rng(seed)
    for i in range(0, X.shape[0]):
        rng.shuffle(miss_mask)
        miss_X[i][miss_mask] = missing_value

    return miss_X


class RandomSampler(object):

    def __init__(self, X, y, seed=None, name=''):
        """ note: use identical seed to obtain the same sampling results across
            different runs
        """

        self.X = X
        self.y = y
        self.rng = np.random.default_rng(seed)
        self.name = name

        # generate labels -> index
        self.y2idx = {}
        for i, l in enumerate(self.y):
            try:
                self.y2idx[l].append(i)
            except KeyError:
                self.y2idx[l] = [i]

    def stratified_random_choice(self, size):

        indexes = []
        for label in self.y2idx:
            indexes.append(self.rng.choice(self.y2idx[label], size=size))

        return np.concatenate(indexes)

# EOF
