from pyramid.config import Configurator
from lkc_bisnp.lib import classifier

def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """
    with Configurator(settings=settings) as config:
        config.include('pyramid_mako')
        config.include('.routes')
        config.scan()

    # prepares the classifiers

    classifier.init_classifiers( config.registry.settings['vvggeo.profiles'])

    return config.make_wsgi_app()
