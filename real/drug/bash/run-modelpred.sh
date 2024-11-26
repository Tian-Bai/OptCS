#!/bin/bash

REGRESSION=$1

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

if [[ "$REGRESSION" == "reg" ]]; then
    datasets=("caco2_wang" "vdss_lombardo" "clearance_microsome_az" "clearance_hepatocyte_az")
    datasets=("ld50_zhu")
    model_list=("DGL_AttentiveFP" "Morgan" "CNN" "rdkit_2d_normalized" "DGL_GCN" "DGL_NeuralFP" "DGL_GIN_AttrMasking" "DGL_GIN_ContextPred" "CNN" "CNN" "CNN" "CNN" "CNN" "CNN" "CNN" "CNN" "CNN")
    conformal_scores=("error" "error" "error" "error" "error" "error" "error" "error" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
else
    datasets=("dili" "herg" "bioavailability_ma" "cyp3a4_substrate_carbonmangels")
    model_list=("DGL_AttentiveFP" "Morgan" "CNN" "rdkit_2d_normalized" "DGL_GCN" "DGL_NeuralFP" "DGL_GIN_AttrMasking" "DGL_GIN_ContextPred")
    conformal_scores=("error" "error" "error" "error" "error" "error" "error" "error")
fi

for data in "${datasets[@]}"; do
    for i in "${!model_list[@]}"; do
        model=${model_list[$i]}
        score=${conformal_scores[$i]}
        alpha=0

        if [[ "$score" != "error" ]]; then
            alpha=$score
            score="quantile"
        fi

        # Assemble slurm order for this job
        JOBN="modelpred:"$data",model:"$model",score":$score

        # Submit the job
        SCRIPT="submit-modelpred.sh $data $model $score 1 $alpha"
            
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