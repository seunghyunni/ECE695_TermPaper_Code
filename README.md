# ECE695_TermPaper_Code
This repository is the implementation for training and testing the experiment presented in the term paper of the class ECE 695.


The experiment is run on dataset from MICCAI 2017 ACDC challenge. Preprocessed version of the subset of the dataset is already prepared inside the 'data_preprocessed' folder. To access and download the full dataset, please refer to (https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTraining.html).


## Requirements

To install requirements:

go to 'scripts' folder and execute following lines.
```setup
bash install_deps.sh
bash conda_deps.sh
```

## Dataset

The preprocessed data files are prepared in the 'data_preprocessed' folder. 

## Training

To train the model, run this command:

```train
python train.py
```

Optimal parameters used for training are already hard-coded in the train.py script.

## Testing

To test our model, load the checkpoint of your trained model, and run:

```eval
test.py
```
