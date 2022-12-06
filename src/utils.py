import pandas as pd
import numpy as np
import os
from typing import TypedDict
import json

DEFAULT_OUTPUT_DIR = '../output/'


def predict_error(data_filename: str, output_dir: str = DEFAULT_OUTPUT_DIR, predict_name: str = 'pred', label_name: str = 'label') -> TypedDict(
    'RetType', {'acc_rate': float, 'correct_num': int, 'total_num': int}):
    '''
    calculate prediction error from a file.
    '''
    data = pd.read_json(os.path.join(output_dir, data_filename))
    correct_num = len(data[data[predict_name] == data[label_name]])
    total_num = len(data)
    acc_rate = correct_num / total_num
    return {'acc_rate': acc_rate, 'correct_num': correct_num, 'total_num': total_num}

def dataset_stat(dataset_file: str, groundTruth_name: str = 'groundTruth', candidates_name: str = 'candidates') -> None:
    '''
    print dataset statistics
    '''
    total_num = 0
    idiom_lengths = []
    candidates_nums = []
    blank_nums=[]
    with open(dataset_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            total_num += 1
            groundTruth = data[groundTruth_name]
            candidates = data[candidates_name]
            blank_nums.append(len(groundTruth))
            for idiom in groundTruth:
                idiom_lengths.append(len(idiom))
            for candidate in candidates:
                candidates_nums.append(len(candidate))
                for idiom in candidate:
                    idiom_lengths.append(len(idiom))
    print('summary for dataset: {}'.format(dataset_file))
    print('total_num: ', total_num)
    print('idoim_lengths: ')
    print(pd.Series(idiom_lengths).describe())
    print('candidates_nums: ')
    print(pd.Series(candidates_nums).describe())
    print('blank_nums: ')
    print(pd.Series(blank_nums).describe())
    print('')    