import argparse
import sys 
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', action="store_true", default=False)
parser.add_argument('--host', action="store", default='localhost')
parser.add_argument('--port', action="store", default=6379, type=int)
parser.add_argument('--input-shape', default="NxHxWxC", type=str)
arguments = parser.parse_args()

if arguments.input_shape != "NxHxWxC" and arguments.input_shape != "NxHxWxC":
    print("--input-shape is either NxHxWxC or NxCxHxW. Exiting...")
    sys.exit(1)