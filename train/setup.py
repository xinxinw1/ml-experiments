import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('tag')

args = parser.parse_args()
tag = args.tag
