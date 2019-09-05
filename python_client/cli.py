import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', action="store_true", default=False)
parser.add_argument('--host', action="store", default='localhost')
parser.add_argument('--port', action="store", default=6379, type=int)

arguments = parser.parse_args()
