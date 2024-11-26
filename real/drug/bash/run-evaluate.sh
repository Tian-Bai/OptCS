#!/bin/bash

# Slurm parameters
PART=hns,stat,owners,normal,candes  # Partition names
MEMO=25G                     # Memory required (50G)
TIME=00-12:00:00             # Time required (1d)
CORE=10                      # Cores required (10)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" -n 1 -c "$CORE" -p "$PART" --time="$TIME" --chdir=/scratch/users/ying531/opt-cs/drug/"

# Create directory for log files
LOGS=logs
mkdir -p $LOGS

# update pythonpath
export PYTHONPATH=/home/users/ying531/.local/lib/python3.12/site-packages:$PYTHONPATH

classification_dataset=("dili" "herg" "bioavailability_ma" "cyp3a4_substrate_carbonmangels" "bbb_martins" "cyp2d6_substrate_carbonmangels" "cyp2c9_substrate_carbonmangels" "hia_hou")
regression_dataset=("caco2_wang" "vdss_lombardo" "clearance_microsome_az" "clearance_hepatocyte_az")
regression_dataset=("lipophilicity_astrazeneca" "ppbr_az" "ld50_zhu" "half_life_obach")

for dataset in "${regression_dataset[@]}"; do
    for q in `seq 0.05 0.05 0.30`; do
        # Assemble slurm order for this job
        JOBN="eval:"$dataset",q:"$q

        # Submit the job
        SCRIPT="submit-evaluate.sh $dataset $q"
            
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

# for dataset in "${classification_dataset[@]}"; do
#     for q in `seq 0.05 0.05 0.30`; do
#         # Assemble slurm order for this job
#         JOBN="eval:"$dataset",q:"$q

#         # Submit the job
#         SCRIPT="submit-evaluate.sh $dataset $q"
            
#         OUTF=$LOGS"/"$JOBN".out"
#         ERRF=$LOGS"/"$JOBN".err"

#         # Assemble slurm order for this job
#         ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
#         # ORD="scancel -n "$JOBN  

#         # Print order
#         echo $ORD

#         # Submit order
#         $ORD
#     done
# done