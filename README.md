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





# urdu_semantic_learning
