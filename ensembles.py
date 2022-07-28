"""
This script is used for competitions.
It ensembles .csv submissions
by averaging files, with a weight for each file.
The average method can be either arithmeitc or geometric.

====== Instructions ======
1) create a directory called: submissions_to_ensemble
2) put the submissions' CSV files into that directory
3) run this code and select the weight of each submission
4) the new CSV will be generated (ensemble_submission.csv)
"""

import os, glob, re
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

path = 'submissions_to_ensemble' # to be modified according to your directory, 
                                 # e.g. set it os.getcwd() if .csv files are in current directory

# load data into a dict, <(str)filename, dataframe>
data = { f.split('/')[-1]: \
        pd.read_csv(f) for f in glob.glob(path + '/*.csv') }

# an alternative option to read data, using regular expression
#data = { re.search(r'([0-9A-Za-z._-]*?.csv)', f).group():
#        pd.read_csv(f) for f in glob.glob(path + '/*.csv') }
assert(len(data) > 0)
print('Loaded files:', list(data.keys()))

def ensemble(data, w, method='arithmetic'):
    """
    @params: data: a dict of dataframes, <(str)filename: dataframe>
             w: a dict of weights, <(str)filename: (int or float)weight>
             method: either arithmetic mean or geometric mean.
    @return: a new dataframe for submission
    """
    columns = data[list(data.keys())[0]].columns
    submission = pd.DataFrame({columns[0]: data[list(data.keys())[0]][columns[0]]})
    assert(method in ['arithmetic', 'geometric'])
    
    if method == 'arithmetic':
        submission[columns[1]] = 0.0
        for key in data.keys():
            submission[columns[1]] += data[key][columns[1]] * w[key]
        submission[columns[1]] /= sum(w.values())
    else:
        submission[columns[1]] = 1.0
        for key in data.keys():
            submission[columns[1]] *= data[key][columns[1]] ** w[key]
        submission[columns[1]] **= 1. / sum(w.values())

    # transform into binary labels
    mask = submission[columns[1]] >= 0.5
    submission[columns[1]][mask] = 1
    submission[columns[1]][~mask] = 0
    submission = submission.astype({columns[1]: 'int32'})
    
    return submission

# Enter weights here
w = { key: 1 for key in data.keys() }
for key in w:
    w[key] = float(input("Enter the weight for {}: ".format(key)))

print('\nWeights for each file:', w)

filename = 'ensemble_submission.csv'
new_submission = ensemble(data, w)
new_submission.to_csv(filename, index=False)
print('New submission file {} is now created'.format(filename))