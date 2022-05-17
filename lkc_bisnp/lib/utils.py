# utility

import sys


def cerr(text):
    print(text, file=sys.stderr)


def cexit(text):
    cerr(text)
    sys.exit(1)

# EOF
