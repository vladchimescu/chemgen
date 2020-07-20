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
