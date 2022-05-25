#
import yaml
import joblib as jlib
import pandas as pd
import html

from pyramid.httpexceptions import HTTPInternalServerError

CLASSIFIERS_ = {}


def init_classifiers(pathname):
    global CLASSIFIERS_
    d = yaml.load(open(pathname + '/classifiers.yaml'), Loader=yaml.Loader)

    for idx, clf_spec in enumerate(d['classifiers']):
        (vectorizer, clf) = jlib.load(pathname + '/' + clf_spec['file'])
        CLASSIFIERS_[idx] = (clf_spec['code'], vectorizer, clf)

    return CLASSIFIERS_


def get_classifier(idx):
    global CLASSIFIERS_
    return CLASSIFIERS_[int(idx)]


def get_all_classifiers():
    global CLASSIFIERS_
    return CLASSIFIERS_


class Literal(object):
    def __init__(self, s):
        self.s = s

    def __html__(self):
        return self.s


data_formats = (
    ('txt-bts', "Text with snp_barcode<TAB>sample_id per line"),
    ('txt-stb', "Text with sample_id<TAB>snp_barcode per line"),
    ('csv', "CSV/TSV ~ comma- or tab-separated value with first column for sample_id and CHROM:POS header"),
)


def preprocess_input(instream, data_fmt):
    # convert data to 0-1-2 values
    try:
        match data_fmt:
            case 'txt-bts':
                data = pd.read_table(instream, sep='\t', header=None)
                sample_ids = data.iloc[:, 1].to_list()
                barcodes = data.iloc[:, 0].to_list()
            case 'txt-stb':
                data = pd.read_table(instream, sep='\t', header=None)
                sample_ids = data.iloc[:, 0].to_list()
                barcodes = data.iloc[:, 1].to_list()         
            case 'csv' | 'tsv':
                data = pd.read_table(instream, sep=None, engine='python')
                sample_ids = data.iloc[:, 0]
                barcodes = data.iloc[:, 1:]
            case _:
                pass

        return sample_ids, barcodes

    except IndexError:
        instream.seek(0)
        lines = [line for line in instream]
        raise HTTPInternalServerError(
            Literal(
                "Error in parsing input. Please check data format.<br><br>" +
                html.escape(f"Selected data format: {dict(data_formats)[data_fmt]}") + "<br><br>" +
                "First 2 line of input: <br><br>" +
                "<br><br>".join(html.escape(line.decode('UTF-8')) for line in lines[:2])
            )
        )

# EOF
