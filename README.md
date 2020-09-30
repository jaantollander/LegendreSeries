# Legendre Series for Piecewise Analytic Functions
This repository is related to my article [*Exploring the Pointwise Convergence of Legendre Series for Piecewise Analytic Functions*](https://jaantollander.com/post/exploring-the-pointwise-convergence-of-legendre-series-for-piecewise-analytic-functions/).

## Contents
- `legendre_series.py` contains the code for computing the Legendre series and analysing the pointwise convergence.
- `plots.py` contains the code for plotting the results from the computations.
- `scripts` directory contains script files for creating the different plots.


## Instructions
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda).

Install Conda environment
```bash
conda env create -f environment.yml 
```

Activate the environment
```bash
source activate legendre36
```

Create the figures
```bash
python scripts/run_legendre_polynomials.py
python scripts/run_piecewise_functions.py
python scripts/run_legendre_series.py
python scripts/run_pointwise_convergence.py
python scripts/run_convergence_distance.py
```

Additionally, creating the animations requires [FFmpeg](https://www.ffmpeg.org/).
```bash
python scripts/run_animation.py
```
