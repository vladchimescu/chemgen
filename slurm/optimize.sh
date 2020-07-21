#!/bin/bash
                                                                                                                                 
#SBATCH -t 01:00:00                                                             
#SBATCH -n 1 
#SBATCH -p htc                                                                   
#SBATCH --mem=16G                                                                                                    
#SBATCH --output=/g/typas/Personal_Folders/bassler/chem_gen/logs/slurm.out

source activate my-mofa-env

script=$1
Xdata=$2
echo "Input file X: " $Xdata
ydata=$3
echo "Input file y: " $ydata
clf=$4
echo "Classifier: " $clf
idx=$5
echo "Grid index: " $idx
python3 $script $Xdata $ydata $clf $idx
