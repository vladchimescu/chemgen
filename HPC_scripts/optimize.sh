#!/bin/bash

#SBATCH -N 1                                                                    
#SBATCH -p htc                                                                  
#SBATCH -t 24:00:00                                                             
#SBATCH -n 1                                                                    
#SBATCH --mem=16G                                                               
#SBATCH --mail-user=vkim@embl.de                                     
#SBATCH --output=logs/slurm-%A.out 
#SBATCH --mail-type=FAIL 

source /g/funcgen/gbcs/miniconda2/bin/deactivate
unset PYTHONPATH
source /g/funcgen/gbcs/miniconda2/bin/activate xgbenv

script=$1
Xdata=$2
echo "Input file X: " $Xdata
ydata=$3
echo "Input file y: " $ydata
clf=$4
echo "Classifier: " $clf
idx=$5
echo "Grid index: " $idx
genes=$6
python $script $Xdata $ydata $clf $idx $genes
