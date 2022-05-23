
import argparse
import numpy as np
import pandas as pd
import joblib as jlib

from lkc_bisnp.lib.utils import cerr
from lkc_bisnp.lib.reader import GenotypeVectorizer, read_barcode
from lkc_bisnp.lib.classifier import BialleleLK, NaNBernoulliNB, CalibratedClassifierCVExt
from lkc_bisnp.lib.metrics import prepare_stratified_samples


def init_argparser(p=None):

    if not p:
        p = argparse.ArgumentParser('train.py')

    p.add_argument('--posfile', required=True)
    p.add_argument('--classfile', default=None)
    p.add_argument('--method', default='balk', choices=['balk', 'bnb', 'iso-balk'])
    p.add_argument('-o', '--outfile', default='classifier.joblib.gz')
    p.add_argument('infile')
    return p


ALLELE_TYPE = 2


def train(args):

    # read position file as pandas dataframe
    positions = pd.read_table(args.posfile)

    # create vectorizer instance
    vectorizer = GenotypeVectorizer(positions, allele_type=ALLELE_TYPE)

    # read training barcode
    train_barcodes, train_Y = read_barcode(args.infile, read_class=args.classfile is None)
    train_data = vectorizer.vectorize(train_barcodes)

    # read training class file and replace train_Y, if necessary
    if args.classfile:
        train_Y = []
        with open(args.classfile) as fin:
            for line in fin:
                train_Y.append(line.strip())
        train_Y = np.array(train_Y)

    cv = 1
    match args.method:
        case 'balk':
            clf = BialleleLK()
        case 'bnb':
            clf = NaNBernoulliNB()
        case 'iso-balk':
            clf = CalibratedClassifierCVExt(BialleleLK(), cv=5, method='isotonic')
            cv = 5
        case _:
            raise ValueError(f'unknown method: {args.method}')

    X, y = prepare_stratified_samples(train_data.X, train_Y, cv)
    clf.fit(X, y)

    jlib.dump((vectorizer, clf), args.outfile)
    cerr(f'[INFO - writing classifier to {args.outfile}]')


def main(args):
    train(args)

# EOF
