#!/bin/bash

# Slurm parameters
PART=hns,stat,owners,normal,candes  # Partition names
MEMO=25G                     # Memory required (50G)
TIME=00-12:00:00             # Time required (1d)
CORE=40                      # Cores required (10)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" -n 1 -c "$CORE" -p "$PART" --time="$TIME" --chdir=/scratch/users/ying531/opt-cs/simulation/"

# Create directory for log files
LOGS=logs
mkdir -p $LOGS

# update pythonpath
export PYTHONPATH=/home/users/ying531/.local/lib/python3.12/site-packages:$PYTHONPATH

ntrain=100
ncalib=100
ntest=100
sigma=1
dim=20
report_indiv=0
Nrep=10

for seed in {1..50}; do
    # Assemble slurm order for this job
    JOBN="SSJ"$setting","$seed

    # Submit the job
    SCRIPT="submit-c_Jin.sh $ntrain $ncalib $ntest $sigma $dim $report_indiv $Nrep $seed"
        
    OUTF=$LOGS"/"$JOBN".out"
    ERRF=$LOGS"/"$JOBN".err"

    # Assemble slurm order for this job
    ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
    # ORD="scancel -n "$JOBN  

    # Print order
    echo $ORD

    # Submit order
    $ORD
done