# Vision Classifiers
COL828 Assignment 1, Sem 1, 2025-26

### Setting up the environment

```
conda create -n vis_env python==3.10
conda activate vis_env
pip install -r requirements.txt
```
### Running the experiments

The code assumes that the dataset is present in the root directory in the following structures:

```
orthonet/
    orthonet data/
        orthonet data/...
    test.csv
    train_split.csv
    train.csv
    val_split.csv
```

```
pacemakers/
    Test/
        BIO - Actros_Philos/ (images…)
        BIO - Cyclos/ (images…)
        BIO - Evia/ (images…)
        BOS - Altrua_Insignia/ (images...)
        ...
```


Simply run the following command to run the experiment as described:

```
python <test/train>_<subtask>.py
```