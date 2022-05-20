
import argparse
import os
import pandas as pd
import joblib as jlib
from lkc_bisnp.lib.utils import cerr
from lkc_bisnp.lib.reader import read_barcode


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
    cerr(f'[INFO - loading classifier from {args.jlibfile}]')
    (vectorizer, clf) = jlib.load(args.jlibfile)

    # preprocess input barcode and perform prediction
    input_data = vectorizer.vectorize(input_barcodes)

    # print marker information
    cerr(f'[INFO - markers used: {len(input_data.markers)}]')
    for marker_id in input_data.markers:
        cerr(f'[    - {marker_id}]')

    preds, proba = clf.predict_proba_partition(input_data.X, 3)

    pred_df = pd.DataFrame(
        {'Prediction_1': preds[:, 0], 'Prob_1': proba[:, 0],
         'Prediction_2': preds[:, 1], 'Prob_2': proba[:, 1],
         'Prediction_3': preds[:, 2], 'Prob_3': proba[:, 2],
         'Count': input_data.count,
         'Hets': input_data.hets,
         'Miss': input_data.miss,
         'Mask': input_data.get_mask_sums(),
         'Notes': input_data.get_logs()}
    )

    if args.outfile:
        pred_df.to_csv(args.outfile, index=False,
                       sep=',' if os.path.splitext(args.outfile) == '.csv' else '\t')
        cerr(f'[INFO - prediction written to {args.outfile}]')
    else:
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(pred_df)


def main(args):
    predict(args)

# EOF
