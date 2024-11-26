#!/bin/bash

# Slurm parameters
PART=hns,stat,owners,normal,candes  # Partition names
MEMO=25G                     # Memory required (50G)
TIME=00-12:00:00             # Time required (1d)
CORE=40                      # Cores required (10)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" -n 1 -c "$CORE" -p "$PART" --time="$TIME" --chdir=/scratch/users/ying531/opt-cs/llm/"

# Create directory for log files
LOGS=logs
mkdir -p $LOGS

# update pythonpath
export PYTHONPATH=/home/users/ying531/.local/lib/python3.9/site-packages:$PYTHONPATH

groupN=5

for idx in {0..99}; do
    # Assemble slurm order for this job
    JOBN="llm1_"$idx

    # Submit the job
    SCRIPT="submit-llm_set1.sh $idx $groupN"
        
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