#!/bin/bash

#SBATCH -N 1                                                                    
#SBATCH -p gpu                                                                  
#SBATCH -t 00:45:00                                                             
#SBATCH -n 1                                                                    
#SBATCH --mem=8G                                                               
#SBATCH --mail-user=stefan.bassler@embl.de                                     
#SBATCH --output=logs/slurm-%A.out 
#SBATCH --mail-type=FAIL 

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
python $script $Xdata $ydata $clf $idx
