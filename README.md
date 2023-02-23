# Road Segmentation

This project is done as part of the [Computational Intelligence Lab](https://www.vorlesungen.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?semkez=2022S&ansicht=KATALOGDATEN&lerneinheitId=157374&lang=en) course offered at the Computer Science department of ETH Zurich.

See the [Kaggle competition](https://www.kaggle.com/competitions/cil-road-segmentation-2022/leaderboard) where our team was ranked top 5.

<img src="https://github.com/jjurm/road-segmentation/raw/master/assets/unet-eval.png" width="600" height="190" />

## Report

Read the associated *report* below, or see the [report repository](https://github.com/jjurm/road-segmentation-report).

<a href="https://github.com/jjurm/road-segmentation/blob/master/road-segmentation.pdf" target="_blank"><img src="https://github.com/jjurm/road-segmentation/raw/master/assets/road-segmentation-01.png" width="300" height="424" /></a>

## Team

- **Felix Sarnthein** ([@safelix](https://github.com/safelix))
- **Rares Constantin** ([@raresionut1](https://github.com/raresionut1))
- **Juraj Micko** ([@jjurm](https://github.com/jjurm))
- **Aashish Kumar Singh** ([@aashishksingh](https://github.com/aashishksingh))

## Abstract

*The task of image segmentation has been widely explored, and a range of algorithms find a wide domain of usage nowadays. Satellites produce a massive amount of high-quality images across all landscapes and potentially enable machine map generation. However, human labeled data is usually expensive to produce and labels might be noisy. Road Segmentation is the problem of dividing each image into regions that contain roads and regions that do not. We explore patch-wise and pixel-wise approaches to road segmentation in a small dataset with noisy labels.*

## Baseline

- download and put the dataset into `../cil-road-segmentation-2022/` (with directories `test` and `training` insite), or change the `PATH_PREFIX` env variable (see the beginning of the notebook)


## Setup

1. Find newest python environment (`module available`) and load it environment: 
`module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy`
2. Create new virtual environment, I'd recommend like so:
`python -m venv $HOME/venv/cil`
3. Activate your virtual environment:
`source venv venv/cil/bin/activate`
4. Install requirements for cluster:
`pip install -r requirements_cluster.txt`
5. Download the data into scratch.
```
cd $SCRATCH
git clone git@github.com:safelix/cil-data.git
```
5. I recommend setting up the .bashrc for cil project:
```
module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy
source venv venv/cil/bin/activate
export CIL_DATA=$SCRATCH/cil-data
export CIL_RESULTS=$HOME/cil/results
```
6. Log in to W&B: `wandb login`


## Submit Jobs
My goal would be to store more general configs in a config/ directory and to change them with commandline parameters. For now I just run the two baseline models with defaults.

`bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python train.py --model LinearConv --model_out pixel --loss_in pixel`

`bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python train.py --model LinearConv --model_out pixel --loss_in patch`

`bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python train.py --model LinearConv --model_out patch --loss_in patch`

`bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python train.py --model LinearFC --model_out patch --loss_in patch`




