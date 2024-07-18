import yaml
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os

from dataset import OAD_Dataset
from yaml.loader import FullLoader
from torch.utils.data import DataLoader
from model import OADModel
from utils import get_4state_proposals, get_hungarian_score
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='yamls/canonical.yaml')
    parser.add_argument('--json_name', type=str, default='canonical.json')
    parser.add_argument('--load_model', type=str, default='checkpoint.pt')
    args = parser.parse_args()
    yaml_path = args.yaml_path
    json_name = args.json_name
    load_model = args.load_model
    with open(yaml_path, encoding='utf-8') as f:
        config = yaml.load(f, FullLoader)
    #Agent
    agent = OADModel(config)
    agent.load_state_dict(torch.load(os.path.join(config['model_save_path'], load_model), map_location=config['device']))
    agent.eval()
    agent.to(config['device'])
    #Dataset
    test_dataset = OAD_Dataset(config, mode='test')
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    results = {}
    cnt = 0
    tp_cum = 0
    p_cum = 0
    a_cum = 0
    with open(config['oracle_proposal_path'], 'rb') as f:
        oracle_proposal_dict = pickle.load(f)
    for feature, video_name in tqdm(testloader):
        video_name = video_name[0]
        atsumari = []
        feature = feature.squeeze().to(config['device'])
        duration = len(feature)
        hs = [torch.zeros(1, config['n_recurrent_hidden']).to(config['device']) for _ in range(config['n_recurrent_layer'])]
        for i in range(duration):
            with torch.no_grad():
                snippet = feature[i].unsqueeze(0)
                hs, score = agent.encode(snippet, hs)
                a = torch.argmax(score, dim=1).squeeze().cpu().numpy().item()
                atsumari.append(a)
        oracle_proposals = oracle_proposal_dict[video_name]
        pred_proposals = get_4state_proposals(atsumari)
        hungarian_results = get_hungarian_score(oracle_proposals, pred_proposals)
        tp_cum += hungarian_results['tp']
        p_cum += hungarian_results['p']
        a_cum += hungarian_results['a']
        l = []
        if len(pred_proposals) == 0:
            pred_proposals.append([0,0])
        for instance in pred_proposals:
            tmp = dict()
            tmp["segment"] = instance
            l.append(tmp)
        results[video_name] = l
    precision = tp_cum/(p_cum+1e-8)
    recall = tp_cum/(a_cum+1e-8)
    f1 = (2*precision*recall)/(precision+recall+1e-8)
    #print(f'hungarian f1, recall: {f1}, {recall}')
    final_dict = dict()
    final_dict['results'] = results
    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f)
