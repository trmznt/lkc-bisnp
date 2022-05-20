
import argparse
import numpy as np
import joblib as jlib
from lkc_bisnp.lib.utils import cerr
from lkc_bisnp.lib.reader import read_barcode


def init_argparser(p=None):

    if not p:
        p = argparse.ArgumentParser('predict.py')

    p.add_argument('--jlibfile', default='classifier.joblib.gz')
    p.add_argument('-o', '--outfile', required=True)
    p.add_argument('infile')
    return p


def vectorize(args):

    # read input file
    input_barcodes, samples = read_barcode(args.infile)

    # load classifier
    cerr(f'[INFO - loading classifier from {args.jlibfile}]')
    (vectorizer, clf) = jlib.load(args.jlibfile)

    # preprocess input barcode and perform prediction
    input_X, input_mask, logs = vectorizer.vectorize(input_barcodes)

    if args.outfile:
        np.savetxt(args.outfile, input_X)
        cerr(f'[INFO - vector is written to {args.outfile}]')


def main(args):
    vectorize(args)

# EOF
