
import time
import argparse
import os
import numpy as np
import pandas as pd
import joblib as jlib

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, make_scorer

from lkc_bisnp.lib.utils import cerr
from lkc_bisnp.lib.reader import GenotypeVectorizer, read_barcode
from lkc_bisnp.lib.classifier import BialleleLK, NaNBernoulliNB, CalibratedClassifierCVExt
from lkc_bisnp.lib.metrics import cross_validate, cross_val_predict


def init_argparser(p=None):

    if not p:
        p = argparse.ArgumentParser('train.py')

    p.add_argument('-o', '--outfile', default='crossval-results.txt')
    p.add_argument('-k', '--kfold', type=int, default=3)
    p.add_argument('-r', '--repeats', type=int, default=10)
    p.add_argument('-t', '--threads', type=int, default=1)
    p.add_argument('-p', '--prediction', default=False, action='store_true',
                   help='save prediction results instead of scores')
    p.add_argument('infile')
    # p.add_argument('infile')
    return p


def crossval(args):

    cerr(f'[INFO - reading model file: {args.infile}]')

    models = eval(open(args.infile).read())
    cerr(f'[INFO - {len(models)} model(s) created]')

    random_seed = int(time.time())
    aggregate_scores = []

    for (code, clf, barcode_file, class_file, pos_file, allele_type) in models:

        cerr(f'[INFO - cross-validating model {code}]')
        positions = pd.read_table(pos_file)
        vectorizer = GenotypeVectorizer(positions, allele_type=allele_type)

        # prepare X and y for this model
        barcodes, _ = read_barcode(barcode_file)

        # read training class file and replace train_Y, if necessary
        Y = []
        with open(class_file) as fin:
            for line in fin:
                Y.append(line.strip())
        Y = np.array(Y)

        # vectorize barcodes
        data = vectorizer.vectorize(barcodes)

        rskf = RepeatedStratifiedKFold(n_splits=args.kfold,
                                       n_repeats=args.repeats,
                                       random_state=random_seed)

        if args.prediction:

            results = cross_val_predict(clf, data.X, Y, cv=rskf, n_jobs=args.threads)

        else:

            results = cross_validate(clf, data.X, Y, cv=rskf, n_jobs=args.threads)

        results['MODEL'] = code
        aggregate_scores.append(results)

    # write aggregate_scores by combining into single dataframe
    df = pd.concat(aggregate_scores)
    df.to_csv(
        args.outfile,
        sep=',' if os.path.splitext(args.outfile)[0] == '.csv' else '\t', index=False
    )
    cerr(f'[INFO - results written to {args.outfile}]')


def main(args):
    crossval(args)

# EOF
