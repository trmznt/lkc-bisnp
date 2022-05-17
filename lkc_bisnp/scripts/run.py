# run.py

import sys
import os
import argparse
import importlib

from lkc_bisnp.lib.utils import cerr, cexit

PATHS = ['lkc_bisnp.scripts.']


def load_module(command):

    if command.endswith('.py') or '/' in command:
        module_name = 'SCRIPT'

        spec = importlib.util.spec_from_file_location(module_name, command)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    M = None
    for module_path in PATHS:
        try:
            M = importlib.import_module(module_path + command)
            cerr('Importing <%s> from path: %s' % (command, module_path))
            break
        except ImportError as exc:
            raise
    if M is None:
        cexit('Cannot locate script name: %s' % command)

    return M


def execute(scriptname, run_args):

    M = load_module(scriptname)
    if hasattr(M, 'init_argparser'):
        parser = M.init_argparser()
        assert parser, "FATAL ERROR - init_argparser() does not return an instance"
        parser = arg_parser(parser=parser)
    else:
        cerr('[WARN - Using default arg parser]')
        parser = arg_parser()

    args = parser.parse_args(run_args)

    cerr('Running module: %s' % scriptname)
    if hasattr(args, 'debug') and args.debug:
        from ipdb import launch_ipdb_on_exception
        with launch_ipdb_on_exception():
            M.main(args)
    else:
        M.main(args)


def greet():
    cerr(f'{sys.argv[0]} - lkc_bisnp script utility')


def usage():
    cexit(f'Usage:\n\t{sys.argv[0]} scriptname [options]')


def arg_parser(description=None, parser=None):

    if not parser:
        parser = argparse.ArgumentParser(description=description, conflict_handler='resolve')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='launch ipdb for handling exception')
    return parser


def main():

    args = sys.argv

    greet()

    if len(args) <= 1:
        usage()

    scriptname = args[1]
    execute(scriptname, args[2:])


if __name__ == '__main__':
    print('This script must be run as lkc_bisnp-run')

# EOF
