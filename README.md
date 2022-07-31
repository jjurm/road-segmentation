# CIL Project 2022: Road Segmentation

## Setup

### Data
**original**: Download and put the provided dataset into $SCRATCH/cil-data (with subdirectories `training` and `test` inside). Manually split the `training ` set into 2 sets: `training_split_train` and `training_split_val`. We have utilized a 9:1 ratio.
**additional**: For the additional dataset, containing 12,000 images gathered from Google Maps, we have a separate, private repository (to avoid data leakage for following competitions). If needed for reproducing results, you can request an invitation.

1. Find newest python environment (`module available`) and load it: 
`module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy`
2. Create new virtual environment, we would recommend like so:
`python -m venv $HOME/venv/cil`
3. Activate the virtual environment:
`source venv venv/cil/bin/activate`
4. Install requirements for cluster:
`pip install -r requirements_cluster.txt`
5. Download the data into scratch.
```
cd $SCRATCH
git clone git@github.com:safelix/cil-data.git (the private data repo)
```
5. We recommend setting up the .bashrc for this project:
```
module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy
source venv venv/cil/bin/activate
export CIL_DATA=$SCRATCH/cil-data
export CIL_RESULTS=$HOME/cil/results
```
6. Log in to W&B: `wandb login`


## Submit Jobs
After the setup is finished, the baseline models can be run using the following commands:

Patch-wise U-Net: `bsub -n 4 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python train.py --model BaselineUNet --model_out patch --loss_in patch --n_epochs 250 --train_dir training_split_train --valid_dir training_split_val --run_name unet_patch`

Pixel-wise U-Net: `bsub -n 4 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python train.py --model BaselineUNet --model_out pixel --loss_in pixel --n_epochs 250 --train_dir training_split_train --valid_dir training_split_val --run_name unet_pixel`

All our relevant experiments, as well as the commands used to run them can be found in the `experiments_overview.xlsx` file.

For creating model ensembles as described in our report, you can add the .csv predictions into a directory called `submissions_to_ensemble` and use the `ensembles.py` file. 



