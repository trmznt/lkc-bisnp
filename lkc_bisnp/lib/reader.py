
import os
import pandas as pd
import numpy as np

from lkc_bisnp.lib.utils import cerr, cexit


def type_0(x):
    return {'N': 1, 'X': 0, x['REF']: 0, x['ALT']: 1}


def type_1(x):
    return {'N': np.nan, 'X': np.nan, x['REF']: 0, x['ALT']: 1}


def type_2(x):
    return {'N': 1, 'X': np.nan, x['REF']: 0, x['ALT']: 2}


class GenotypeVectorizer(object):
    """ this class will vectorize a SNP barcode (a string of SNP alleles)
        into a numpy array suitable for machine learning inputs
    """

    def __init__(self, notations, allele_type=1):
        """ notations is a list of tuple (chrom, pos, ref, alt) or a
            dataframe of (CHROM, POS, REF, ALT)
        """

        if not isinstance(notations, pd.DataFrame):
            self.notations = pd.DataFrame(notations)
        else:
            self.notations = notations
        self.allele_type = allele_type
        self.dicts = self._create_dicts()

    def _create_dicts(self):
        dicts = []

        match self.allele_type:
            case 0:
                f = type_0
            case 1:
                f = type_1
            case 2:
                f = type_2
            case _:
                raise ValueError(f'type {self.allele_type} is not defined')

        for idx, row in self.notations.iterrows():
            dicts.append(f(row))

        return dicts

    def vectorize_string(self, barcodes):
        """ return a numpy array of translated barcodes, where barcodes is
            a list of string of SNP alleles, eg: "ATCGNTCXTCA" with
            N as hets and X as missing data
        """

        logs = []
        arr = np.zeros((len(barcodes), len(barcodes[0])))

        # mask is True if the position is missing or undefined (eg. other alt alleles)
        masks = np.full((len(barcodes), len(barcodes[0])), False)

        for i, s in enumerate(barcodes):
            if len(s) != len(self.notations):
                logs.append(f'ERR: requires {len(self.notations)} SNPs, receives {len(s)} SNPs')
                continue
            for j, a in enumerate(s):
                try:
                    arr[i][j] = self.dicts[j][a]
                except KeyError:
                    arr[i][j] = 0
                    masks[i][j] = True
            if masks[i].sum() > 0:
                logs.append(f'WARN: {masks[i].sum()} SNPs were masked')
            else:
                logs.append('')

        return arr, masks, logs

    def vectorize_dataframe(self, barcodes):
        """ return a numpy array of translated barcodes from dataframe, where columns
            have labels in format of chrom:pos, eg: PvP01_01_v1:253612
        """

        arr = np.zeros((len(barcodes), len(barcodes.columns)))
        indexes = []

        # sane checking and reindex columns
        for c in barcodes.columns:
            match c.split(':'):
                case [chrom, pos]:
                    pos = int(pos)
                    position = self.notations[(self.notations['CHROM'] == chrom) & 
                                              (self.notations['POS'] == pos)]
                    indexes = position.index


                case _:
                    raise ValueError(f'Column name is not supported: f{c}')

    def vectorize(self, barcodes):

        match barcodes:
            case pd.DataFrame():
                return self.vectorize_dataframe(barcodes)
            case list() | np.ndarray():
                return self.vectorize_string(barcodes)
            case _:
                raise ValueError(f'Unsupported barcode type: {type(barcodes)}')

        return None


def read_string(filename, read_class=False):
    barcodes = []
    class_list = []
    with open(filename) as fin:
        for line in fin:
            tokens = line.strip().split()
            barcodes.append(tokens[1].upper())
            if read_class:
                class_list.append(tokens[0])

    cerr(f'[I - reading {len(barcodes)} barcode(s) from {filename}]')
    return np.array(barcodes), np.array(class_list)


def read_dataframe(filename, read_class=False):
    df = pd.read_table(filename, sep=None)


def read_barcode(filename, read_class=False):

    match os.path.splitext(filename)[1].lower():
        case '.txt':
            return read_string(filename, read_class)
        case '.csv' | '.tsv':
            return read_dataframe(filename, read_class)
        case _:
            raise ValueError('undefined')

    return None, None

# EOF
