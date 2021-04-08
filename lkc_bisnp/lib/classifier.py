import pickle
import numpy
from .lkest import SNPProfile, SNPLikelihoodEstimator

import logging
log = logging.getLogger(__name__)

# put the initial data here

_CLASSIFIERS_ = {}

def set_classifiers(classifiers):

    global CLASSIFIERS
    _CLASSIFIERS_ = classifiers


def get_classifiers():
    return _CLASSIFIERS_


def init_classifiers(profilepath):

    with open(profilepath, 'rb') as f:
        profile_data = pickle.load( f )

    _classifiers_ = get_classifiers()

    for code in profile_data:
        profile = SNPProfile.from_dict( profile_data[code] )
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
                'Sample {} had incorrect {} SNPs instead of the required {} SNPs'.format(
                        sample_id, len(barcode), len(positions))
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
        sample_ids.append( sample_id)
        haplotypes.append( haplotype )
        if mask > 0:
            logs.append( 'Sample {} was masked for {} SNPs'.format( sample_id, masks) )

    haplotypes = numpy.array(haplotypes, dtype=numpy.int8)
    return (haplotypes, sample_ids, logs)


def classify(haplotypes):
    pass


