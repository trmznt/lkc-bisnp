from pyramid.config import Configurator
from lkc_bisnp.web.lib.utils import init_classifiers

def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """
    with Configurator(settings=settings) as config:
        config.include('pyramid_mako')
        config.include('.routes')
        config.scan()

    # prepares the classifiers

    init_classifiers(config.registry.settings['vvggeo.datadir'])

    return config.make_wsgi_app()

# EOF
