"""Fresh-process execution of a complete, explicit stage recipe."""
import argparse
import os
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['n', 'a_v1', 'a_v2'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    os.environ['BTP_ROOT'] = str(args.root.resolve())
    runpy.run_path(str(Path(__file__).parent / 'recipes' / (args.stage + '.py')), run_name='__main__')


if __name__ == '__main__':
    main()
