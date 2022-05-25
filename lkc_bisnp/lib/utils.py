# utility

import sys


def cerr(text):
    print(text, file=sys.stderr)


def cout(text):
    print(text)


def cexit(text):
    cerr(text)
    sys.exit(1)

# EOF
