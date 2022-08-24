
import time
import argparse
import os
import numpy as np
import pandas as pd
import joblib as jlib

from lkc_bisnp.lib.utils import cerr
from lkc_bisnp.lib.reader import GenotypeVectorizer, read_barcode
from lkc_bisnp.lib.metrics import RandomSampler, create_missingness, prepare_dataframe_metrics, unique_labels

from lkc_bisnp.lib.classifier import BialleleLK, NaNBernoulliNB, CalibratedClassifierCVExt


def init_argparser(p=None):

    if not p:
        p = argparse.ArgumentParser('simulate.py')

    p.add_argument('-o', '--outfile', default='simulation-results.txt')
    p.add_argument('-r', '--repeats', type=int, default=10)
    p.add_argument('-s', '--samplesize', type=int, default=25,
                   help='number of sample per class')
    p.add_argument('-t', '--threads', type=int, default=1)
    p.add_argument('--missingness', default=None)
    p.add_argument('infile')

    return p


def simulate(args):

    cerr(f'[INFO - reading model file: {args.infile}]')

    models = eval(open(args.infile).read())
    cerr(f'[INFO - {len(models)} model(s) created]')

    random_seed = int(time.time())
    aggregate_scores = []

    missingness_params = None
    if args.missingness:
        missingness_params = [float(x) for x in args.missingness.split(',')]

    for (code, clf, barcode_file, class_file, pos_file, allele_type) in models:

        cerr(f'[INFO - simulating model {code}]')
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

        # train the classifier
        clf.fit(data.X, Y)

        random_sampler = RandomSampler(data.X, Y, seed=random_seed, name=code)

        for repeat in range(args.repeats):

            # create simulation sets

            sim_indexes = random_sampler.stratified_random_choice(args.samplesize)
            sim_X = data.X[sim_indexes]
            sim_Y = Y[sim_indexes]

            if missingness_params is not None:

                for p in missingness_params:

                    miss_X = create_missingness(sim_X, p)
                    y_pred = clf.predict(miss_X)
                    labels = unique_labels(sim_Y)

                    scores = prepare_dataframe_metrics(sim_Y, y_pred, labels, fold_id=repeat)
                    scores['MISSINGNESS'] = p
                    scores['MODEL'] = code

                    aggregate_scores.append(scores)

            else:
                raise ValueError('this section should not be run!')

    # write aggregate_scores by combining into single dataframe
    df = pd.concat(aggregate_scores)
    df.to_csv(
        args.outfile,
        sep=',' if os.path.splitext(args.outfile)[0] == '.csv' else '\t', index=False
    )
    cerr(f'[INFO - results written to {args.outfile}]')


def main(args):
    simulate(args)

# EOF
