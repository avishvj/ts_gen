from __future__ import print_function
import os, sys, time, random
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from rdkit import Chem, Geometry
from optparse import OptionParser

# mlflow
import mlflow
import mlflow.sklearn

# syspath for current directory
sys.path.insert(0, "../model/")
from G2C import G2C


def eval_metrics(actual, pred):
        return

def read_in_data(r_file, ts_file, p_file):
        return

def prepare_batch(batch_mols):
        return



if __name__ == "__main__":

        # read in args

        # funcs: read in data files and prepare batches

        # build experiment base path

        # tensorflow inside mlflow scope?
        #       - dataset
        #       - build model
        #       - launch session and do
        #       - print and/or save metrics + best models
        #       - save D_init, W, GNN embeddings
        # 



        # functions to read in the data, split

        # functions to 


parser = OptionParser()

parser.add_option("--restore", dest="restore", default=None)
parser.add_option("--layers", dest="layers", default=2)
parser.add_option("--hidden_size", dest="hidden_size", default=128)
parser.add_option("--iterations", dest="iterations", default=3)
parser.add_option("--epochs", dest="epochs", default=200)
parser.add_option("--batch_size", dest="batch_size", default=8)
parser.add_option("--gpu", dest="gpu", default=0)

parser.add_option("--r_file", dest="r_file", default=None)
parser.add_option("--p_file", dest="p_file", default=None)
parser.add_option("--ts_file", dest="ts_file", default=None)

args, _ = parser.parse_args()
layers = int(args.layers)
hidden_size = int(args.hidden_size)
iterations = int(args.iterations)
gpu = str(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

reactantFile = str(args.r_file)
tsFile = str(args.p_file)
productFile = str(args.ts_file)

QUEUE = True
BATCH_SIZE = int(args.batch_size)
EPOCHS = int(args.epochs)
best_val_loss = 9e99 # set to max

# Load dataset
print("Loading datset")
start = time.time()
data = [Chem.SDMolSupplier(reactantFile, removeHs=False, sanitize=False),
        Chem.SDMolSupplier(tsFile, removeHs=False, sanitize=False),
        Chem.SDMolSupplier(productFile, removeHs=False, sanitize=False)]
data = [(x,y,z) for (x,y,z) in zip(data[0],data[1],data[2]) if (x,y,z)]

elapsed = time.time() - start
print(" ... loaded {} molecules in {:.2f}s".format(len(data), elapsed))

# Dataset specific dimensions
elements = "HCNO"
num_elements = len(elements)
max_size = max([x.GetNumAtoms() for x,y,z in data])
print(max_size)

# need to adapt train.py to train model how I want



# funcs:
#   - eval_metrics()
#   - "__main__" with mlflow.start_run()
#       - run model, fit train data, predict qualities, predict model params
#       - got some mlflow log metrics

# mlflow
#   - mlflow tracking: record and query experiments: code, data, config, results
#       - parameters (key-value inputs to your code), metrics (numeric values that update over time), tags and notes (info about a run), artifacts (files, data, models), source (what code ran?), version, run (mlflow code instance), experiment ({run, .., run})
#   - mlflow projects: package code in format that enables reproducible runs on many platforms
#   - api-first: submit runs, log models, metrics, etc. 