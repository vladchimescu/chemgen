# Conservation of Drug Interactions Across Bacterial Species
Using the combination screen data we can perform higher-level analysis of conserved interactions on strain and species level. 

In order to run conservation analysis and quality check on the whole gram-positive combinatorial screen simply run:

```sbatch run_cons.sh /g/huber/users/vkim/gitlab/parallel```


In addition to conservation, we can apply network inference to perform clustering and characterize the interactome of each strain.


Another important objective is to investigate the underlying mechanism of interactions and predict drug-drug interactions using single-drug chemical genetics data available for 3 gram-negaive species (*E. coli*, *S. enterica*, *P. aeruginosa*)


## Prediction of Drug-Drug Interactions (Synergies / Antagonisms)
The repository has a number of Jupyter notebooks in which prediction of drug-drug interactions is explored in great detail. The notebooks are in the subdirectory `Jupyter/`.


Furthermore Python and Slurm scripts are available for performing predictions using high-performance computing. The class for storing and cross-validating interactions is in `Python/predictor.py`. Some useful functions are in `Python/chemgen_utils.py`. 

Prediction / classification of synergies and antagonisms is based either on
+ single-compound chemical genetics data in *E. coli* and *Salmonella* (`Python/chemgen-prdict.py`)
+ structural motifs / molecular fingerprints such as circular ECPF4 fingerprints (`Python/struct-predict.py`)


In order to launch predictions on EMBL cluster:


```bash bashscripts/submit_prediction.sh```

For `RandomForestClassifier` one can plot OOB error rate as a function of forest size (`n_estimators`). On EMBL cluster


```bash bashscripts/get_oob_rate.sh```


## Hyperparameter Optimization
For brute-force grid search (on a defined grid) run


```
bash bashscripts/grid_search.sh Python/find_optimal_hyperparams.py \
data/chemgenetics/nichols_binarized.csv data/chemgenetics/nichols_y.csv randomforest
```


which will generate weighted average precision for each parameter grid value and store it in data/optimization subdirectory.

For Salmonella chemical genetics data:

``` bash
bash bashscripts/grid_search.sh Python/find_optimal_hyperparams.py \
data/chemgenetics/salmonella_binarized.csv data/chemgenetics/salmonella_y.csv randomforest
```

Similarly for structural predictions in E. coli, Salmonella and Pseudomonas:

``` bash
bash bashscripts/grid_search.sh Python/find_optimal_struct_hyperparams.py \
data/chemicals/drugSmiles.txt data/chemicals/ecoli-traintest.csv randomforest

bash bashscripts/grid_search.sh Python/find_optimal_struct_hyperparams.py \
data/chemicals/drugSmiles.txt data/chemicals/salmonella-traintest.csv randomforest

bash bashscripts/grid_search.sh Python/find_optimal_struct_hyperparams.py \
data/chemicals/drugSmiles.txt data/chemicals/pseudomonas-traintest.csv randomforest
```


For One-vs-Rest classifier hyperparameter tuning run the followng scripts (chemical genetics):
```
# E. coli (Nichols data)
bash bashscripts/grid_search.sh Python/multiclass_hyperparams.py \
data/chemgenetics/nichols_binarized.csv data/chemgenetics/nichols_y.csv randomforest

# Salmonella
bash bashscripts/grid_search.sh Python/multiclass_hyperparams.py \
data/chemgenetics/salmonella_binarized.csv data/chemgenetics/salmonella_y.csv randomforest
```

Similarly to find the best hyperparameters for One-vs-Rest classifier based on molecular fingerprints run:

``` bash
bash bashscripts/grid_search.sh Python/multiclass_struct_params.py \
data/chemicals/drugSmiles.txt data/chemicals/ecoli-traintest.csv randomforest

bash bashscripts/grid_search.sh Python/multiclass_struct_params.py \
data/chemicals/drugSmiles.txt data/chemicals/salmonella-traintest.csv randomforest

bash bashscripts/grid_search.sh Python/multiclass_struct_params.py \
data/chemicals/drugSmiles.txt data/chemicals/pseudomonas-traintest.csv randomforest

```
