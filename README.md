# Antibiotic Synergy and Antagonism Prediction Based On Chemical Genetics
The repository has a number of Jupyter notebooks in which prediction of drug-drug interactions is explored in great detail. The notebooks are in the subdirectory `Jupyter/`.


Furthermore Python and Slurm scripts are available for performing predictions using high-performance computing. The class for storing and cross-validating interactions is in `Python/predictor.py`. Some useful functions are in `Python/chemgen_utils.py`. 

Prediction / classification of synergies and antagonisms is based on single-compound chemical genetics data in *E. coli* and *Salmonella*.


## Hyperparameter Optimization
We used grid search on a pre-defined parameter grid to find optimal model hyperparameters. For One-vs-Rest classifier hyperparameter tuning run the followng scripts (chemical genetics):
```
# E. coli (Nichols data)
bash HPC_scripts/grid_search.sh Python/multiclass_hyperparams.py \
data/chemgenetics/nichols_signed.csv data/chemgenetics/nichols_y.csv randomforest

# Salmonella
bash HPC_scripts/grid_search.sh Python/multiclass_hyperparams.py \
data/chemgenetics/salmonella_signed.csv data/chemgenetics/salmonella_y.csv randomforest
```

In order to run XGBoost:
```
# E. coli (Nichols data)
bash HPC_scripts/grid_search.sh Python/multiclass_hyperparams.py \
data/chemgenetics/nichols_signed.csv data/chemgenetics/nichols_y.csv xgboost

# Salmonella
bash HPC_scripts/grid_search.sh Python/multiclass_hyperparams.py \
data/chemgenetics/salmonella_signed.csv data/chemgenetics/salmonella_y.csv xgboost
```
