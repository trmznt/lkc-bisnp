#
import yaml
import joblib as jlib

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

# EOF
