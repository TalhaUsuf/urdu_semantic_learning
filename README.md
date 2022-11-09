# How to use the DVC
```bash
# Initialize
$ dvc init

# Track data directory
$ dvc add data # Create data.dvc
$ git add data.dvc
$ git commit -m "add data"

# Store the data remotely
$ dvc remote add -d remote gdrive://lynNBbT-4J0ida0eKYQqZZbC93juUUUbVH

# Push the data to remote storage
$ dvc push 

# Get the data
$ dvc pull 

# Switch between different version
$ git checkout HEAD^1 data.dvc
$ dvc checkout

```

# CUDA path set 📁

This works with environment `yoloface` conda environment.
 
CUDA path status is shown below:
 - torch ✔
 - tensorflow ❌

To use tensorflow inside this environment 🐍 , write following line in _bash_
```bash
export LD_LIBRARY_PATH=/home/talha/anaconda3/envs/yoloface/lib/:$LD_LIBRARY_PATH
```
# Datasets


1. mswc_microset
   1. The intent of this small subset is to aid in preliminary experimentation, inspection, and tutorials, without requiring users to download the full MSWC dataset or the full subset of MSWC in English or Spanish.
2. multilingual_context_73_0.8011
   1. A multilingual embedding model, which is a pretrained keyword feature extractor which can be used to perform few-shot keyword spotting. A multilingual embedding model, which is a pretrained keyword feature extractor which can be used to perform few-shot keyword spotting.
3. speech_commands
   1. a reference-quality keyword spotting dataset in English. We will use the background noise samples in GSC for finetuning our model.
4. unknown_files.tar.gz
   1. An unknown-keyword dataset: a precomputed bank of unknown keywords to preserve the ability for a few-shot model to distinguish between the target keyword and non-target keywords.

```html
finetuning_datasets/
├── mswc_microset
├── mswc_microset.tar.gz
├── mswc_microset.tar.gz.dvc
├── multilingual_context_73_0.8011
├── multilingual_context_73_0.8011.tar.gz
├── multilingual_context_73_0.8011.tar.gz.dvc
├── speech_commands
├── speech_commands_v0.02.tar.gz
├── speech_commands_v0.02.tar.gz.dvc
├── unknown_files.tar.gz
├── unknown_files.tar.gz.dvc
├── unknown_files.txt
└── unknown_words
```




# urdu_semantic_learning



# Spectrogram samples

Model needs a spectrogram as input, spectrogram shape : `49, 40`

| **Spectrogram Attribute**  |  **value** |
|:---------------------------|-----------:|
| time steps                 |         49 |
| freq. bins                 |         40 |



![](images/spectrogram.png)




