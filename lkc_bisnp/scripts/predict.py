
import argparse
import numpy as np
import pandas as pd
import joblib as jlib
from lkc_bisnp.lib.utils import cerr
from lkc_bisnp.lib.reader import GenotypeVectorizer, read_barcode
from lkc_bisnp.lib.classifier import BialleleLK


def init_argparser(p=None):

    if not p:
        p = argparse.ArgumentParser('predict.py')

    p.add_argument('--jlibfile', default='classifier.joblib.gz')
    p.add_argument('-o', '--outfile', default=None)
    p.add_argument('infile')
    return p


def predict(args):

    # read input file
    input_barcodes, _ = read_barcode(args.infile)

    # load classifier
    (vectorizer, clf) = jlib.load(args.jlibfile)

    # preprocess input barcode and perform prediction
    input_X, input_mask = vectorizer.vectorize(input_barcodes)
    preds, proba = clf.predict_proba_partition(input_X, 3)

    pred_df = pd.DataFrame(
        {'Prediction_1': preds[:, 0], 'CalibratedProb_1': proba[:, 0],
         'Prediction_2': preds[:, 1], 'CalibratedProb_2': proba[:, 1],
         'Prediction_3': preds[:, 2], 'CalibratedProb_3': proba[:, 2]}
    )

    if args.outfile:
        pred_df.to_csv(args.outfile, index=False)
        cerr(f'[INFO - prediction written to {args.outfile}]')
    else:
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(pred_df)


def main(args):
    predict(args)

# EOF
