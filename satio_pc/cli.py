import sys
import argparse


def extract_s2(cli_args):
    from satio_pc.extraction import S2BlockExtractor
    parser = argparse.ArgumentParser()
    parser.add_argument('tile')
    parser.add_argument('block_id', type=int)
    parser.add_argument('year', type=int)
    parser.add_argument('-o', '--output', default='.')
    parser.add_argument('-k', '--connstr')
    parser.add_argument('-r', '--container')
    parser.add_argument('-c', '--cleanup', action='store_true')
    parser.add_argument('-t', '--terminate', action='store_true')

    args = parser.parse_args(cli_args)

    s2ex = S2BlockExtractor(args.tile,
                            args.block_id,
                            args.year,
                            output_folder=args.output,
                            connection_str=args.connstr,
                            container_name=args.container,
                            cleanup=args.cleanup,
                            terminate_if_failed=args.terminate)

    s2ex.extract()


class Ewc:

    def __init__(self, cli_args=None):

        self.args = cli_args or sys.argv[1:]

        parser = argparse.ArgumentParser("ESA WorldCover processing CLI")
        parser.add_argument('command', help="Commands: 'l2a' or 'gamma0'")

        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(self.args[:1])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def l2a(self):
        extract_s2(cli_args=self.args[1:])

    def gamma0(self):
        # process_bounds_cli(cli_args=self.args[1:])
        ...


def ewc_cli():
    Ewc()


if __name__ == '__main__':
    ewc_cli()
