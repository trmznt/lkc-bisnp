from pyramid.view import view_config

from lkc_bisnp.web.lib.utils import get_all_classifiers


@view_config(route_name='home', renderer='../templates/home.mako')
def home(request):
    classifiers = get_all_classifiers()
    return {'project': 'vivaxgen-barcodes', 'classifiers': classifiers}

# EOF
