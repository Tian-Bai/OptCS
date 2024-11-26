#!/bin/bash 

# limit sklearn parallelism
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export MKL_NUM_THREADS=3
export VECLIB_MAXIMUM_THREADS=3
export NUMEXPR_NUM_THREADS=3

#now run normal batch commands


ml python/3.9.0
ml py-numpy/1.24.2_py39
ml py-pandas/2.0.1_py39
ml py-scikit-learn/1.3.2_py39
ml py-scipy/1.10.1_py39

## NO user serviceable part below
for i in $(seq 1 $2); do
    python3 llm_set1.py --seedgroup $(($2 * $1 + i)) --repN 1 &
done

wait