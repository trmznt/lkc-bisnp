
import io

import pandas as pd

from pyramid.view import view_config
from pyramid.httpexceptions import HTTPFound

from lkc_bisnp.web.lib.utils import get_classifier


@view_config(route_name='run', renderer='../templates/run.mako')
def run(request):

    # get data from form
    if request.method != 'POST':
        return HTTPFound(location='/')

    barcode_data = request.POST.get('BarcodeData', '')
    input_file = request.POST.get('InFile', None)
    data_fmt = request.POST.get('DataFormat', 'txt-bts')    # default to Barcode-tab-Sample
    clf_idx = request.POST.get('Classifier', '')

    if input_file is None or input_file == b'':
        instream = io.StringIO(barcode_data)
    else:
        instream = input_file.file
    
    code, vectorizer, classifier = get_classifier(clf_idx)

    # convert data to 0-1-2 values
    match data_fmt:
        case 'txt-bts':
            data = [line.strip().split('\t') for line in instream if line.strip()]
            barcodes = [x[0] for x in data]
            sample_ids = [x[1] for x in data]
        case 'txt-stb':
            data = [line.strip().split('\t') for line in instream if line.strip()]
            barcodes = [x[1] for x in data]
            sample_ids = [x[0] for x in data]           
        case 'csv' | 'tsv':
            data = pd.read_table(instream, sep=None, engine='python')
            sample_ids = data.iloc[:, 0]
            barcodes = data.iloc[:, 1:]
        case _:
            pass


    data = vectorizer.vectorize(barcodes)

    predictions, probas = classifier.predict_proba_partition(data.X, 3)

    table = []
    for sample_id, prediction, proba, hets, miss, masked, log in zip(sample_ids, predictions, probas,
                                                                     data.hets, data.miss, data.get_mask_sums(),
                                                                     data.get_logs()):
        if log.startswith('ERR'):
            proba = [-1] * len(proba)
        table.append([sample_id] +
                     ['[{:5.3f}] {}'.format(prob, pred) if prob > 0.01 else '-' for prob, pred in zip(proba, prediction)] +
                     [hets, miss, masked, log]
                     )

    return {'table': table, 'code': code}

    raise RuntimeError

    # perform classification

    # return results

    results = None

    return {'results': results}
