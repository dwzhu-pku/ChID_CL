# this file contains the functions to process the data from dictionary

import numpy as np
import pandas as pd
import json
import os
import re
import copy
from tqdm import tqdm

DEFAULT_RAW_PATH = '../dataset/dicts-raw/'
OUTPUT_PATH = '../dataset/'

dict1_path = os.path.join(DEFAULT_RAW_PATH, 'chinese-idioms-12976.txt')
dict2_path = os.path.join(DEFAULT_RAW_PATH, 'xinhua-idiom.json')

output1_path = os.path.join(OUTPUT_PATH, 'idiom-dict1.json')
output2_path = os.path.join(OUTPUT_PATH, 'idiom-dict2.json')

groundTruth_tag = 'groundTruth'
candidates_tag = 'candidates'
content_tag = 'content'
realCount_tag = 'realCount'
candidate_num = 7


def produce_samples(dict_df: pd.DataFrame, output_path: str):
    print(
        f'producing samples from {dict_df.shape[0]} idioms, saving to {output_path}')
    # get all the idioms
    idioms = dict_df['word'].values
    # only use idioms with length == 4
    idioms = idioms[np.array([len(idiom) == 4 for idiom in idioms])]
    # print(idioms)

    # create a list of samples
    samples_from_dict = []
    for i in tqdm(range(len(idioms))):
        idiom = dict_df['word'][i]
        sample = {}
        sample[groundTruth_tag] = [idiom]

        # from meaning to get the candidates
        if not pd.isnull(dict_df['explanation'][i]) and len(dict_df['explanation'][i]) > 1:
            cands = np.random.choice(
                idioms[idioms != idiom], candidate_num-1, replace=False).tolist()
            cands.append(idiom)
            np.random.shuffle(cands)
            sample[candidates_tag] = [cands]

            content = f'#idiom#：{dict_df["explanation"][i]}'
            sample[content_tag] = content
            sample[realCount_tag] = 1
            samples_from_dict.append(copy.deepcopy(sample))

        # from example to get the candidates
        if not pd.isnull(dict_df['example'][i]):
            cands = np.random.choice(
                idioms[idioms != idiom], candidate_num-1, replace=False).tolist()
            cands.append(idiom)
            np.random.shuffle(cands)
            sample[candidates_tag] = [cands]

            content = dict_df['example'][i].replace('～', '#idiom#')
            if '#idiom#' in content:
                sample[content_tag] = content
                sample[realCount_tag] = 1
                samples_from_dict.append(copy.deepcopy(sample))

        # from derivation to get the candidates
        if not pd.isnull(dict_df['derivation'][i]):
            cands = np.random.choice(
                idioms[idioms != idiom], candidate_num-1, replace=False).tolist()
            cands.append(idiom)
            np.random.shuffle(cands)
            sample[candidates_tag] = [cands]

            content = dict_df['derivation'][i].replace(idiom, '#idiom#')
            if '#idiom#' in content:
                sample[content_tag] = content
                sample[realCount_tag] = 1
                samples_from_dict.append(copy.deepcopy(sample))

    # save the samples
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples_from_dict:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print('samples produced and saved!')


# process the data from dict1
dict1 = pd.read_csv(dict1_path, sep=',', header=None)
dict1.columns = ['id', 'word', 'pinyin',
                 'explanation', 'derivation', 'example', 'abbreviation']
# print(dict1.head())
produce_samples(dict1, output1_path)

# process the data from dict2
dict2 = pd.read_json(dict2_path, orient='records', encoding='utf-8')
# print(dict2.head())
produce_samples(dict2, output2_path)
