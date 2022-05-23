#
# classifier classes

import pickle
import math
import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.extmath import safe_sparse_dot

from .lkest import SNPProfile, SNPLikelihoodEstimator

import logging
log = logging.getLogger(__name__)


def cerr(text):
    print(text, file=sys.stderr)


def cout(text):
    print(text)


class _MethodExt(object):

    def predict_proba_partition(self, X, kth=1):
        """ return (predictions, probabilitiies) """

        proba = self.predict_proba(X)

        # perform argsort to get the indexes for ascending values,
        # get the kth greatest values, and reverse the indexes
        classes_indexes = np.argsort(proba, axis=1)[:, -kth:][:, ::-1]

        # take classes_ and proba based on classes_indexes
        predictions = self.classes_[classes_indexes]
        probabilities = np.take_along_axis(proba, classes_indexes, axis=1)
        return predictions, probabilities


class NaNBinomialNB(BernoulliNB, _MethodExt):
    """ BernoulliNB with missing data capability """

    binomial_n_ = None
    log_binomial_coeff_ = None

    def __init__(self, drop_coeff=True, **kwargs):
        super().__init__(**kwargs)
        self.drop_coeff = drop_coeff

    def _init_counters(self, n_classes, n_features):
        super()._init_counters(n_classes, n_features)
        self.data_count_ = np.zeros((n_classes, n_features), dtype=np.int32)
        if not self.drop_coeff:
            self.log_binomial_coeff_ = np.log(
                np.array([math.comb(self.binomial_n_, k)
                         for k in range(0, self.binomial_n_ + 1)])
            )

    def _check_X(self, X):
        # allow for NaN (missing values)
        return self._validate_data(X, accept_sparse='csr', reset=False, force_all_finite=False)

    def _check_X_y(self, X, y, reset=True):
        # allow for NaN (missing values)
        return self._validate_data(X, y, accept_sparse="csr", reset=reset, force_all_finite=False)

    def _count(self, X, Y, data_count_multiplier=1):
        """ X ~ array of feature per sample, eg:
                [   [0, 1, NaN, 0, 1, 0, 0, NaN, NaN, 0],
                    [0, 0, 0, NaN, 0, 0, 0, 0, 0, 1],
                    ...
                ]
            Y ~ array of class per sample, eg with 4 class
                [   [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    ...
                ]
        """
        # use BernoulliNB._count() but convert NaN to zero
        super()._count(np.nan_to_num(X), Y)

        # instead of class_count_, data_count_ holds number of samples per feature
        self.data_count_ += safe_sparse_dot(
            Y.T, (~np.isnan(X)).astype(int) * self.binomial_n_)

    def _update_feature_log_prob(self, alpha):
        smoothed_fc = self.feature_count_ + alpha
        smoothed_dc = self.data_count_ + alpha * 2

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_dc)

    def _joint_log_likelihood(self, X):
        """ calculate log likelihood of samples X """
        n_features = self.feature_log_prob_.shape[1]
        n_features_X = X.shape[1]

        if n_features_X != n_features:
            raise ValueError(
                "Expected input with %d features, got %d instead"
                % (n_features, n_features_X)
            )

        # to allow for skipping NaN (missing value)
        X = np.ma.masked_array(X, mask=np.isnan(X))

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute  ∑(log_coeff + x · flp + (1 - x) · neg_prob)

        jll = np.ma.dot(X, self.feature_log_prob_.T) + np.ma.dot((self.binomial_n_ - X), neg_prob.T)

        # broadcast log_coeff if we have log_binomial_coeff_, otherwise just drop coeff
        if self.log_binomial_coeff_ is not None:
            jll += np.array([self.log_binomial_coeff_[np.nan_to_num(X, 0).astype(int)].sum(axis=1)]).T

        if self.fit_prior is True:
            jll += self.class_log_prior_

        return jll.filled()


class NaNBernoulliNB(NaNBinomialNB):

    binomial_n_ = 1


class BialleleLK(NaNBinomialNB):
    """
        SNP data will be treated as diploid dataset, and input data is
        the count number of alternate allele of respective SNP data, hence
        any occurence of alternate allele will be counted as 2 allele.
        Example:

            ref:    ATCGAT
            alt:    TAGCTA

            seq:    ATCGAT
            data:   000000

            seq:    AAGCAT
            data:   022200

            seq:    ANCGAT
            data:   010000

            seq:    AT-CAT
            data:   00!200  (! = NaN)

        The formula to calculate likelihood is:

            Lk(X|C_k) = SUM (p_ki)^(x_i) (1 - p_ki)^(2-x_i)

        where:
            X ~ feature sets (SNP input data)
            C_k ~ class k
            p_ki = frequency of alternate allele at position i of class k
            x_i ~ number of alternate allele at position i

        feature_log_prob: log of proportion of alternate allele at
                          at position i of class k
    """

    # use NaNBinomialNB with N=2
    binomial_n_ = 2

    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha, fit_prior=False)


class CalibratedClassifierCVExt(CalibratedClassifierCV, _MethodExt):
    pass


# put the initial data here

_CLASSIFIERS_ = {}


def set_classifiers(classifiers):

    global CLASSIFIERS
    _CLASSIFIERS_ = classifiers


def get_classifiers():
    return _CLASSIFIERS_


def init_classifiers(profilepath):

    with open(profilepath, 'rb') as f:
        profile_data = pickle.load(f)

    _classifiers_ = get_classifiers()

    for code in profile_data:
        profile = SNPProfile.from_dict(profile_data[code])
        _classifiers_[code] = SNPLikelihoodEstimator(profile)

    log.info('Classifiers initialized from profile {}'.format(profilepath))


def prepare_data(datalines, snpset):
    sample_ids = []
    haplotypes = []
    logs = []

    classifier = get_classifiers()[snpset]
    positions = classifier.get_profile().positions

    for idx, tokens in enumerate(datalines):
        barcode = tokens[0]
        sample_id = tokens[1]

        if len(barcode) != len(positions):
            logs.append(
                f'Sample {sample_id} had incorrect {len(barcode)} SNPs instead of '
                'the required {len(positions)} SNPs'
            )
            continue

        # check barcode
        barcode = barcode.upper()
        haplotype = []
        hets = []
        masks = []
        mask = 0
        for allel, position in zip(barcode, classifier.get_profile().positions):
            if allel in ['X', 'N']:
                haplotype.append(1)
            elif allel == position[2]:
                haplotype.append(0)
            elif allel == position[3]:
                haplotype.append(2)
            else:
                haplotype.append(1)
                mask += 1
        sample_ids.append(sample_id)
        haplotypes.append(haplotype)
        if mask > 0:
            logs.append(f'Sample {sample_id} was masked for {mask} SNPs')

    haplotypes = numpy.array(haplotypes, dtype=numpy.int8)
    return (haplotypes, sample_ids, logs)


def classify(haplotypes):
    pass

# EOF
