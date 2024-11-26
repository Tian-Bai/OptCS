#!/bin/bash

# Slurm parameters
PART=hns,stat,owners,normal,candes  # Partition names
MEMO=25G                     # Memory required (50G)
TIME=02-00:00:00             # Time required (1d)
CORE=20                      # Cores required (20)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" -n 1 -c "$CORE" -p "$PART" --time="$TIME" --chdir=/scratch/users/ying531/opt-cs/simu_loo/"

# Create directory for log files
LOGS=logs
mkdir -p $LOGS

ntest=100
sigma=0.5
dim=10

# use 100 iterations for scaleup
for seed in {1..100}; do
    for nlabel in 50 100 200 300 500 1000; do
        # Assemble slurm order for this job
        # non-selection, simulation, Jin
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
done