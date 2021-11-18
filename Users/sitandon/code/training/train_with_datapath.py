import argparse
import os

parser = argparse.ArgumentParser("train")
parser.add_argument("--input_datapath", type=str, help="sample datapath argument")
parser.add_argument("--synapse_runid", type=str, help="Synapse RunId argument")
args = parser.parse_args()

print("Sample datapath argument: %s" % args.data_path)
print("Sample Synapse RunId argument: %s" % args.synapse_runid)
