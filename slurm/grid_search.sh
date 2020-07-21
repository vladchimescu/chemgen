#!/bin/bash

#SBATCH -N 1                                                                    
#SBATCH -p htc                                                                  
#SBATCH -t 00:30:00                                                             
#SBATCH -n 1                                                                    
#SBATCH --mem=8G                                                                                                              

script=$1
Xdata=$2
ydata=$3
clf=$4

for i in `seq 0 1297`
do
	sbatch bashscripts/optimize.sh $script $Xdata $ydata $clf $i
done
