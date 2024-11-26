#!/bin/bash 

#now run normal batch commands

ml python/3.9.0
ml py-numpy/1.24.2_py39
ml py-pandas/2.0.1_py39
ml py-scikit-learn/1.3.2_py39
ml py-scipy/1.10.1_py39


for modelid in {1..3}; do
    for setting in {1..4}; do
        ## NO user serviceable part below
        python3 loo_Jin.py $1 $2 $modelid $setting $3 $4 $5 &
    done
done

wait