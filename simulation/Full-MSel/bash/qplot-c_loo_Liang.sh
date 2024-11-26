#!/bin/bash

# Slurm parameters
PART=hns,stat,owners,normal,candes  # Partition names
MEMO=25G                     # Memory required (50G)
TIME=01-12:00:00             # Time required (1d)
CORE=20                      # Cores required (20)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" -n 1 -c "$CORE" -p "$PART" --time="$TIME" --chdir=/scratch/users/ying531/opt-cs/simu_loosel/"

# Create directory for log files
LOGS=logs
mkdir -p $LOGS

# update pythonpath
export PYTHONPATH=/home/users/ying531/.local/lib/python3.12/site-packages:$PYTHONPATH

# decrease param for runtime
nlabel=500
ntest=100
sigma=3
dim=100

for seed in {1..100}; do
    # Assemble slurm order for this job
    JOBN="SLSL"$setting","$seed

    # Submit the job
    SCRIPT="submit-c_loo_Liang.sh $nlabel $ntest $seed $sigma $dim"
        
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