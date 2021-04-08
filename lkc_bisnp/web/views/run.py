from pyramid.view import view_config
from pyramid.httpexceptions import HTTPFound

from lkc_bisnp.lib.classifier import get_classifiers, prepare_data


@view_config(route_name='run', renderer='../templates/run.mako')
def run(request):

    # get data from form
    if request.method != 'POST':
        return HTTPFound(location='/')

    barcode_data = request.POST.get('BarcodeData', '')
    snpset = request.POST.get('SNPSet', '')

    data = [ line.strip().split('\t') for line in barcode_data.split('\n') if line.strip()]
    classifier = get_classifiers()[snpset]

    # convert data to 0-1-2 values
    haplotypes, sample_ids, logs = prepare_data(data, snpset)

    predictions, probas = classifier.predict_proba_partition(haplotypes, 3)

    table = []
    for sample_id, prediction, proba in zip(sample_ids, predictions, probas):
        table.append( [sample_id] +
            [ '[{:5.3f}] {}'.format( prob, pred ) if prob > 0.01 else '-' for prob, pred in zip(proba, prediction) ]
            )

    return { 'table': table, 'logs': logs }

    raise RuntimeError

    # perform classification

    # return results

    results = None

    return {'results': results}
