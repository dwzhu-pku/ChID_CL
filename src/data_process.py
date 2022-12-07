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

universe_set_path = os.path.join(OUTPUT_PATH, 'train_data.json')

output1_path = os.path.join(OUTPUT_PATH, 'idiom-dict1.json')
output2_path = os.path.join(OUTPUT_PATH, 'idiom-dict2.json')

groundTruth_tag = 'groundTruth'
candidates_tag = 'candidates'
content_tag = 'content'
realCount_tag = 'realCount'
candidate_num = 7


def produce_samples(dict_df: pd.DataFrame, output_path: str) -> list[dict]:
    print(
        f'producing samples from {dict_df.shape[0]} idioms, saving to {output_path}')
    # get all the idioms
    idioms = dict_df['word'].values
    # only use idioms with length == 4
    idioms = idioms[np.array([len(idiom) == 4 for idiom in idioms])]
    dataset = []
    # only need words in universe set
    with open(universe_set_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    universe_set = []
    for data in dataset:
        for cands in data['candidates']:
            universe_set.extend(cands)
    universe_set =set(universe_set)
    idioms =np.array(list(idiom for idiom in idioms if idiom in universe_set))
    print(idioms)


    # create a list of samples
    samples_from_dict = []
    for i in tqdm(range(len(dict_df))):
        idiom = dict_df['word'][i]
        if idiom not in idioms:
            continue
        sample = {}
        sample[groundTruth_tag] = [idiom]

        # # from meaning to get the candidates
        # if not pd.isnull(dict_df['explanation'][i]) and len(dict_df['explanation'][i]) > 1:
        #     cands = np.random.choice(
        #         idioms[idioms != idiom], candidate_num-1, replace=False).tolist()
        #     cands.append(idiom)
        #     np.random.shuffle(cands)
        #     sample[candidates_tag] = [cands]

        #     content = f'#idiom#：{dict_df["explanation"][i]}'
        #     sample[content_tag] = content
        #     sample[realCount_tag] = 1
        #     samples_from_dict.append(copy.deepcopy(sample))

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

        # # from derivation to get the candidates
        # if not pd.isnull(dict_df['derivation'][i]):
        #     cands = np.random.choice(
        #         idioms[idioms != idiom], candidate_num-1, replace=False).tolist()
        #     cands.append(idiom)
        #     np.random.shuffle(cands)
        #     sample[candidates_tag] = [cands]

        #     content = dict_df['derivation'][i].replace(idiom, '#idiom#')
        #     if '#idiom#' in content:
        #         sample[content_tag] = content
        #         sample[realCount_tag] = 1
        #         samples_from_dict.append(copy.deepcopy(sample))

    # save the samples
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     for sample in samples_from_dict:
    #         f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # print('samples produced and saved!')
    return samples_from_dict

# process the data from dict1
dict1 = pd.read_csv(dict1_path, sep=',', header=None)
dict1.columns = ['id', 'word', 'pinyin',
                 'explanation', 'derivation', 'example', 'abbreviation']
# print(dict1.head())
sample1 = produce_samples(dict1, output1_path)

# process the data from dict2
dict2 = pd.read_json(dict2_path, orient='records', encoding='utf-8')
# print(dict2.head())
sample2 = produce_samples(dict2, output2_path)

# merge the samples, delete the duplicates about groundTruth
samples = sample1 + sample2
groundTruths = [sample[groundTruth_tag][0] for sample in samples]
groundTruths, indices = np.unique(groundTruths, return_index=True)
samples = [samples[i] for i in indices]
# save the samples
with open(os.path.join(OUTPUT_PATH, 'idiom-dict-example.json'), 'w', encoding='utf-8') as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

