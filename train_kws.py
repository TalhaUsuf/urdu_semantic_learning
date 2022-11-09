#%%

# import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import subprocess
import csv
from tqdm import tqdm
# import torch
#%%
datasets_root = Path("finetuning_datasets")

# # convert the samples inside `finetuning_datasets/mswc_microset` to 16kHz
#
# Path(datasets_root, "mswc_microset_16k").mkdir(exist_ok=True, parents=True)
# mswc_original_dataset = datasets_root / "mswc_microset"
# for language in mswc_original_dataset.iterdir():
#         # create a pbar to iterate over word files
#         pbar = tqdm((language / "clips").iterdir(), colour="green")
#         for word in pbar:
#             pbar.set_description(f"Converting {language}/{word.stem}")
#             # make a destination dir. following same dir. structure as original
#             Path(datasets_root, "mswc_microset_16k", language.stem, word.stem).mkdir(parents=True, exist_ok=True)
#             for audio in tqdm(word.iterdir(), colour="magenta"):
#                 destination = Path(datasets_root, "mswc_microset_16k", language.stem, word.stem)
#                 cmd = ["opusdec", "--rate", "16000", audio.as_posix(), f"{destination.as_posix()}/{audio.stem}.wav"]
#                 subprocess.run(cmd)


#%%

# ==============================================================
#                plot some samples as spectrograms
# ==============================================================

from keyword_spotting.input_data import file2spec
from keyword_spotting.input_data import standard_microspeech_model_settings
from rich.console import Console
import matplotlib.pyplot as plt
from pathlib import Path
import random

settings = standard_microspeech_model_settings(label_count=10)
Console().log(f"🔴 spectrogram time steps ---> [red]{settings['spectrogram_length']}")
Console().log(f"🔴 spectrogram freq steps ---> [red]{settings['fingerprint_width']}")

f, ax = plt.subplots(2, 3, figsize=(12, 9))
english_samples = random.sample(list(Path("finetuning_datasets","mswc_microset_16k", "en", "marvin").iterdir()), 6)
ax = ax.flatten()
for k in range(6):
    spectrogram = file2spec(settings, english_samples[k].as_posix())
    ax[k].imshow(spectrogram, cmap='magma')

    ax[k].set_xlabel("Freq.")
    ax[k].set_ylabel("Time")
    ax[k].set_title(f"Spectrogram - marvin-en-{spectrogram.shape}")
plt.tight_layout()
plt.show()
plt.savefig("images/spectrogram.png")

#%%

# ==============================================================
#                       embedding model
# ==============================================================
import tensorflow as tf
model = tf.keras.models.load_model("finetuning_datasets/multilingual_context_73_0.8011")
model.summary()
# remove the last classification layer
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("dense_2").output)

# put model in evaluation mode
model.trainable = False


#%%

# ==============================================================
#                       get embedding from model
# ==============================================================

# get example embedding of 1024 dim.
example = file2spec(settings, english_samples[0].as_posix())
# model.input
# >> <KerasTensor: shape=(None, 49, 40, 1) dtype=float32 (created by layer 'input_1')>
example = example[tf.newaxis, : , : , tf.newaxis] #
embedding_sample = model.predict(example)
Console().log(f"🔴 embedding sample shape ---> [red]{embedding_sample.shape}")
f, ax = plt.subplots(1, 1, figsize=(12, 9))
ax.scatter(range(len(embedding_sample[0])),  embedding_sample[0],
           s=60, edgecolors="k", facecolors="green")
ax.set_xlabel("index")
ax.set_ylabel("Value")
ax.set_title(f"Embedding - marvin-en-{embedding_sample.shape}")
plt.tight_layout()
plt.show()
plt.savefig("images/embedding.png")

#%%
# ==============================================================
#                       train a 5 shot model to classify
#                       spanish keyword 'tiempo'
# ==============================================================
import pandas as pd
splits_csv = Path("finetuning_datasets/mswc_microset/es/es_splits.csv")
df = pd.read_csv(splits_csv.as_posix(), skipinitialspace=True)

Console().log(f"columns are {df.columns}")
TRAIN = df[(df['SET'] == 'TRAIN') & (df['WORD'] == 'tiempo')]
VAL = df[(df['SET'] == 'DEV') & (df['WORD'] == 'tiempo')]
TEST = df[(df['SET'] == 'TEST') & (df['WORD'] == 'tiempo')]

#%%
def change_name(row):
    # Console().print(row)
    ist_part = Path("finetuning_datasets/mswc_microset_16k/es")
    sec_part = Path(row['LINK'])
    name = (ist_part / sec_part).as_posix().split(".opus")[0]+".wav"
    row['PATH'] = name

    return row

#%%

TRAIN_changed = TRAIN.apply(change_name, axis=1)
VAL_changed = VAL.apply(change_name, axis=1)
TEST_changed = TEST.apply(change_name, axis=1)




