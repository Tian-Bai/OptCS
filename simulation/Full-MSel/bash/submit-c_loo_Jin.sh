#!/bin/bash 

# limit sklearn parallelism
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export MKL_NUM_THREADS=3
export VECLIB_MAXIMUM_THREADS=3
export NUMEXPR_NUM_THREADS=3

#now run normal batch commands

ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scikit-learn/1.5.1_py312
ml py-scipy/1.12.0_py312

for setting in {1..4}; do
    ## NO user serviceable part below
    python3 choosescore_loo_Jin.py $1 $2 $setting $3 $4 $5 &
done

wait