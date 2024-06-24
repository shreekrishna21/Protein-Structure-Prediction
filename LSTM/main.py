import tensorflow as tf
import numpy as np
import time
import glob
import sys
import os
from model import MBPredictor

# Random generator initializers
tf.random.set_seed(1)
np.random.seed(1)

if len(sys.argv) != 3:
    print("Usage: python3.4 main.py <train/test> <data_dir>")
    exit()

# Set model parameters
params = {
    'max_seq_len': 41,
    'structures': 5,
    'batch_size': 128,
    'beta': 0.001,
    'lr': 0.0002,
    'num_epochs': 100,
    'hidden_layer_size': 128,
    'stop_check_interval': 50,
    'lstm_cell_size': 64
}

network_name = './network'

# Set data paths
directive = sys.argv[1]
data_dir = sys.argv[2]

# Add trailing dir separator
if not data_dir.endswith(os.path.sep):
    data_dir = data_dir + os.path.sep

struct_files = glob.glob(data_dir + '*annotations*')
sequence_files = glob.glob(data_dir + '*sequences*')

if len(struct_files) == 0 or len(sequence_files) == 0:
    print("Warning: no input files found!")
    exit()

for seq_file, struct_file in zip(sequence_files, struct_files):
    print(seq_file + "\n" + struct_file)
    start_time = time.time()

    data_paths = (seq_file, struct_file)

    # Initialize a LsTM network for motif binding prediction
    net_file = network_name + "-" + seq_file[len(data_dir):]
    loggername = "results-" + os.path.splitext(os.path.basename(__file__))[0] + ".txt"
    predictor = MBPredictor(data_paths, params, net_file, loggername)

    if directive == "train":
        # Hyperparameter optimization
        predictor.train()

    elif directive == "test":
        # Test the model
        result = predictor.test()
        print(result)

    else:
        print("Unknown directive")
        break

    end_time = time.time()
    duration = end_time - start_time
    print(duration)
