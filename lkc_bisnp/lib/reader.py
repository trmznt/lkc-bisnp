
import os
import pandas as pd
import numpy as np

from lkc_bisnp.lib.utils import cerr, cexit


class VectorData(object):

    def __init__(self, X, mask, count, hets, miss, markers):
        self.X = X
        self.mask = mask
        self.count = count
        self.hets = hets
        self.miss = miss
        self.markers = markers
        self.mask_sums = None
        self.logs = None

    def get_logs(self):
        if self.logs is None:
            # create logs
            logs = []
            mask_sums = self.get_mask_sums()
            marker_length = len(self.markers)
            # prepare log by iterating over samples
            for i in range(len(self.X)):
                if self.count[i] < marker_length:
                    logs.append(f'ERR: requires {marker_length} SNPs, receives {self.count[i]} SNPs')
                    continue
                if mask_sums[i] > 0 or self.miss[i] > 0 or self.hets[i] > 0:
                    logs.append(f'WARN: masked: {self.mask[i].sum()}; '
                                f'miss: {self.miss[i]}; hets: {self.hets[i]}')
                else:
                    logs.append('')
            self.logs = logs
        return self.logs

    def get_mask_sums(self):
        if self.mask_sums is None:
            self.mask_sums = self.mask.sum(axis=1)
        return self.mask_sums


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

        # for string data, assume that all positions are used
        markers = [f"{p['CHROM']}:{p['POS']}" for _, p in self.notations.iterrows()]

        arr = np.full((len(barcodes), len(self.notations)), np.nan)

        # mask is True if the position is missing or undefined (eg. other alt alleles)
        mask = np.full(arr.shape, False)

        count = np.zeros(len(barcodes), dtype=int)
        miss = np.zeros(len(barcodes), dtype=int)
        hets = np.zeros(len(barcodes), dtype=int)

        # iterate over samples
        for i, s in enumerate(barcodes):

            miss[i] = s.count('X')
            hets[i] = s.count('N')

            # iterate over positions
            for j, a in enumerate(s):
                count[i] += 1
                try:
                    arr[i][j] = self.dicts[j][a]
                except KeyError:
                    arr[i][j] = np.nan
                    mask[i][j] = True

        return VectorData(X=arr, mask=mask, count=count, hets=hets, miss=miss,
                          markers=markers)

    def vectorize_dataframe(self, barcodes):
        """ return a numpy array of translated barcodes from dataframe, where columns
            have labels in format of chrom:pos, eg: PvP01_01_v1:253612
        """

        # variable to hold index from notation -> barcodes
        indexes = np.full(len(self.notations), -1)

        # prepare indexes
        markers = []
        columns = barcodes.columns.to_list()
        for idx, p in self.notations.iterrows():
            posid = f"{p['CHROM']}:{p['POS']}"
            try:
                barcode_idx = columns.index(posid)
                indexes[idx] = barcode_idx
                markers.append(posid)
            except ValueError:
                pass

        arr = np.full((len(barcodes), len(self.notations)), np.nan)
        mask = np.full(arr.shape, False)
        miss = np.zeros(len(barcodes), dtype=int)
        hets = np.zeros(len(barcodes), dtype=int)
        count = np.zeros(len(barcodes), dtype=int)

        # iterate over columns/positions
        for j in range(len(indexes)):
            # column j in arr is column barcode_idx in barcode

            # skip if this position is not found in columns
            if indexes[j] < 0:
                # add this missing positions to sample miss
                miss += 1
                count += 1
                continue

            # iterate over samples
            for i, a in enumerate(barcodes.iloc[:, indexes[j]]):

                count[i] += 1

                try:
                    arr[i, j] = self.dicts[j][a]
                    match a:
                        case 'N':
                            hets[i] += 1
                        case 'X':
                            miss[i] += 1
                        case _:
                            pass

                except KeyError:
                    if ('|' in a) or ('/' in a):
                        arr[i, j] = self.dicts[j]['N']
                        hets[i] += 1
                    else:
                        arr[i, j] = np.nan
                        mask[i, j] = True
                        cerr(f'Masked for position: {self.notations.iloc[j]} with allel: {a}')
                        #assert 0

        return VectorData(X=arr, mask=mask, count=count, hets=hets, miss=miss,
                          markers=markers)

    def vectorize(self, barcodes):
        """ return (array, mask, logs) """ 

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
    df = pd.read_table(filename, sep=None, engine='python')

    # the first column must be either class_list or sample list
    class_df = df.iloc[:, 0]

    # the rest column are the barcodes
    barcode_df = df.iloc[:, 1:]

    return barcode_df, class_df


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
