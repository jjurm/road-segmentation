# cil-project

## Baseline

- download and put the dataset into `../cil-road-segmentation-2022/` (with directories `test` and `training` insite), or change the `PATH_PREFIX` env variable (see the beginning of the notebook)


## Setup

1. Find newest python environment (`module available`) and load it environment: 
`module load gcc/6.3.0 python_gpu 3.8.5`
2. Create new virtual environment, I'd recommend like so:
`python -m venv $HOME/venv/cil`
3. Activate your virtual environment:
`source venv venv/cil/bin/activate`
4. Install requirements for cluster:
`pip install -r requirements_cluster.txt`
5. I recommend setting up the .bashrc for cil project:
```
module load gcc/6.3.0 python_gpu 3.8.5
source venv venv/cil/bin/activate
export CIL_DATA=$SCRATCH/cil-data
export CIL_RESULTS=$SCRATCH/cil/results
```


## Submit Jobs
My goal would be to store more general configs in a config/ directory and to change them with commandline parameters. For now I just run the two baseline models with defaults.

`bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python train.py --model LinearConv --model_out pixels --loss_in pixels`

`bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python train.py --model LinearConv --model_out pixels --loss_in patches`

`bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python train.py --model LinearConv --model_out patches --loss_in patches`

`bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python train.py --model LinearFC --model_out patches --loss_in patches`




