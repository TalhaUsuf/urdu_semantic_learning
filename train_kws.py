## imports

import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import subprocess
import csv
from tqdm import tqdm
import torch
##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      Google speech commands dataset
#     it is needed to acquire background noise samples
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tf.keras.utils.get_file(origin="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                        untar=True,
                        cache_subdir="speech_commands")


##
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         Multilingual Spoken Words Corpus
#     6000 clips per keyword,.31 keywords in english and 20 in spanish
#   this is needed to experiment with the trained model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tf.keras.utils.get_file(origin="https://storage.googleapis.com/public-datasets-mswc/mswc_microset.tar.gz",
                        untar=True,
                        cache_subdir="mswc_microset")
