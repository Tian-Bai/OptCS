# OptCS

This repository contains the code to reproduce the results in the paper [Optimized Conformal Selection: Powerful Selective Inference After Conformity Score Optimization](https://arxiv.org/abs/2411.17983).

## Folders

- `simulation/`: Simulation experiments
    - `simulation/Msel/`: Conformity score selection with pre-trained models (Section 5.1)
    - `simulation/Full/`: Conformal selection without data splitting (Section 5.2)
    - `simulation/Full-Msel/`: Model training and selection with full data (Section 5.3)
- `real/`: Real-data appliations
    - `real/drug/`: Drug discovery with model selection (Section 6.1)
    - `real/llm/`: Boosting LLM Alignment (Section 6.2)
- `requirements.txt`: A list of required python packages

## Workflow

### Simulation Experiments

For simulation, a single job submission is sufficient to run the experiment.

### Drug Discovery Application

For the drug discovery application:
1. Use `modelpred.py` to generate and save model predictions based on different drug encodings.
2. Evaluate the performance of various methods using `evaluate.py`.

### LLM Alignment Application

The code for the LLM alignment application is largely adapted from [this repository](https://github.com/yugjerry/conformal-alignment). After performing report generation and score extraction as described in the linked repository:
1. Use `collect.py` to compile all uncertainty/confidence scores and labels into a single `.csv` file.
2. Conduct experiments using different combinations of models via `llm_set1.py` and `llm_set2.py`.
