#!/bin/bash

# Slurm parameters
PART=hns,stat,owners,normal,candes  # Partition names
MEMO=25G                     # Memory required (50G)
TIME=00-12:00:00             # Time required (1d)
CORE=20                      # Cores required (20)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" -n 1 -c "$CORE" -p "$PART" --time="$TIME" --chdir=/scratch/users/ying531/opt-cs/simu_loo/"

# Create directory for log files
LOGS=logs
mkdir -p $LOGS

nlabel=200
ntest=100
sigma=0.5
dim=10

for seed in {1..500}; do
    # Assemble slurm order for this job
    JOBN="NSJ"$setting","$seed

    # Submit the job
    SCRIPT="submit-loo_Jin.sh $nlabel $ntest $seed $sigma $dim"
        
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