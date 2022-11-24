import pandas as pd
import numpy as np
import os
from typing import TypedDict

RetType = TypedDict(
    'RetType', {'acc_rate': float, 'correct_num': int, 'total_num': int})
DEFAULT_OUTPUT_DIR = '../output/'


# calculate prediction error from a file
def predict_error(data_filename: str, output_dir: str = DEFAULT_OUTPUT_DIR, predict_name: str = 'pred', label_name: str = 'label') -> RetType:
    data = pd.read_json(os.path.join(output_dir, data_filename))
    correct_num = len(data[data[predict_name] == data[label_name]])
    total_num = len(data)
    acc_rate = correct_num / total_num
    return {'acc_rate': acc_rate, 'correct_num': correct_num, 'total_num': total_num}
