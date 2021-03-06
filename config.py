import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
SAVED_SUMMARIES_DIR = os.path.join(ROOT_DIR, 'saved_summaries')
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')

with open('config.tag.txt') as f:
    TAG = f.read().strip()
