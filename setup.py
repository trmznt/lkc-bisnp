import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    README = f.read()
with open(os.path.join(here, 'CHANGES.md')) as f:
    CHANGES = f.read()

requires = [
    'numpy',
]

web_requires = [
    'plaster_pastedeploy',
    'pyramid',
    'pyramid_mako',
    'pyramid_debugtoolbar',
    'waitress',
]

tests_require = [
    'WebTest >= 1.3.1',  # py3 compat
    'pytest >= 3.7.4',
    'pytest-cov',
]

setup(
    name='lkc_bisnp',
    version='0.0',
    description='lkc_bisnp',
    long_description=README + '\n\n' + CHANGES,
    classifiers=[
        'Programming Language :: Python',
        'Framework :: Pyramid',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    author='Hidayat Trimarsanto',
    author_email='anto@eijkman.go.id',
    url='https://github.com/trmznt/lkc-bisnp',
    keywords='web pyramid pylons',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    extras_require={
        'web': web_requires,
        'testing': tests_require,
    },
    install_requires=requires,
    entry_points={
        'paste.app_factory': [
            'main = lkc_bisnp.web:main',
        ],
    },
)
