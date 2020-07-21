#!/bin/bash
                                                                                                                                 
#SBATCH -t 00:45:00                                                             
#SBATCH -n 1                                                                    
#SBATCH --mem=16G                                                               
#SBATCH --mail-user=stefan.bassler@embl.de                                     
#SBATCH --output=/g/typas/Personal_Folders/bassler/chem_gen/logs/slurm.out
#SBATCH --mail-type=FAIL 

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
