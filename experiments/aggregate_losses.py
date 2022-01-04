
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script with helper functions to aggregate losses stored in log / loss / 

in a tabular format - mainly aggregating over splits (there is a .json for each split)

"""

import sys
import pandas as pd
import numpy as np
from experiments.regression import DATASETS
from utils.config import LOG_DIR
                   
def standard_error(x):
    return np.std(x) / np.sqrt(len(x))

if __name__ == "__main__":
    
    model_name = 'SVGP'

    # Read the results files
    list_of_dict = []
    paths = []
    for d in DATASETS:
        df_list = []
        results_dir = LOG_DIR / "loss"
        loss_files = list(results_dir.glob(f'*{d}*{model_name}*.csv'))
        paths.extend(loss_files)
        df = pd.concat([pd.read_csv(loss_file, header=None) for loss_file in loss_files], axis=1)
        list_of_dict.append({d:df})

    if len(paths) == 0:
        print("No results found - try a different results suffix")
        sys.exit(2)
        
    df_final = pd.DataFrame()    
    for i in np.arange(len(list_of_dict)):
        colname = [str(key) for key, value in list_of_dict[i].items()]
        df_final[colname[0] + '_mean'] = np.mean(list_of_dict[i][colname[0]], axis=1)
        df_final[colname[0] + '_se'] = standard_error(list_of_dict[i][colname[0]].T)

        
    
