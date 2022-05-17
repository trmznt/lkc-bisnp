from pyramid.view import view_config
from pyramid.httpexceptions import HTTPFound

from lkc_bisnp.web.lib.utils import get_classifier


@view_config(route_name='run', renderer='../templates/run.mako')
def run(request):

    # get data from form
    if request.method != 'POST':
        return HTTPFound(location='/')

    barcode_data = request.POST.get('BarcodeData', '')
    clf_idx = request.POST.get('Classifier', '')

    data = [line.strip().split('\t') for line in barcode_data.split('\n') if line.strip()]
    code, vectorizer, classifier = get_classifier(clf_idx)

    # convert data to 0-1-2 values
    barcodes = [x[0] for x in data]
    sample_ids = [x[1] for x in data]
    haplotypes, masks, logs = vectorizer.vectorize(barcodes)

    predictions, probas = classifier.predict_proba_partition(haplotypes, 3)

    table = []
    for sample_id, prediction, proba, log in zip(sample_ids, predictions, probas, logs):
        table.append([sample_id] +
                     ['[{:5.3f}] {}'.format(prob, pred) if prob > 0.01 else '-' for prob, pred in zip(proba, prediction)] +
                     [log]
                     )

    return {'table': table, 'logs': []}

    raise RuntimeError

    # perform classification

    # return results

    results = None

    return {'results': results}
